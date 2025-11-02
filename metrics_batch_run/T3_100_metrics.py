import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Optional
from scipy.optimize import differential_evolution
import random

tf.keras.backend.set_floatx('float64')

# ============================================================
# Small helpers
# ============================================================

class SineActivation(Layer):
    def call(self, inputs):
        return tf.concat([tf.sin(2 * np.pi * inputs), tf.cos(2 * np.pi * inputs)], axis=1)

def make_spd_batch(S: tf.Tensor, min_eig: float = 1e-2) -> tf.Tensor:
    """
    Force a batch of symmetric matrices S (B,3,3) to be SPD by clipping eigenvalues.
    """
    w, v = tf.linalg.eigh(S)                      # (B,3), (B,3,3)
    w = tf.clip_by_value(w, min_eig, np.inf)      # (B,3)
    VWT = tf.matmul(v, v, transpose_b=True)       # (B,3,3)  (just to cache; not strictly needed)
    return tf.matmul(v, tf.matmul(tf.linalg.diag(w), v, transpose_b=True))

# ============================================================
# Metric providers
# ============================================================

class InputDependentRandomMetric:
    """
    Smooth SPD metric g(x) on T^3.
    Coefficients (amplitudes/phases) are random per run, but the metric varies with x.
    """
    def __init__(self, seed: Optional[int] = None):
        self.reseed(seed)

    def reseed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
        # Random amplitudes in a safe range so g stays well-conditioned after SPD projection
        self.a = np.random.uniform(0.1, 0.4, size=6)   # amplitudes for diagonal/off-diagonal
        self.p = np.random.uniform(0, 2*np.pi, size=6) # phases

        # A small isotropic floor added before SPD projection to avoid near-singularity
        self.floor = 0.6

    def tensor(self, x: tf.Tensor) -> tf.Tensor:
        """
        x: (B,3)
        returns g(x): (B,3,3), SPD
        """
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        # periodic features
        s1 = tf.sin(2*np.pi*x[:, 0:1] + self.p[0]); c1 = tf.cos(2*np.pi*x[:, 0:1] + self.p[1])
        s2 = tf.sin(2*np.pi*x[:, 1:2] + self.p[2]); c2 = tf.cos(2*np.pi*x[:, 1:2] + self.p[3])
        s3 = tf.sin(2*np.pi*x[:, 2:3] + self.p[4]); c3 = tf.cos(2*np.pi*x[:, 2:3] + self.p[5])

        # Build a symmetric matrix field H(x)
        h11 = self.floor + self.a[0]*s1 + 0.05*c2
        h22 = self.floor + self.a[1]*s2 + 0.05*c3
        h33 = self.floor + self.a[2]*s3 + 0.05*c1

        h12 = 0.15*(s1*s2) + 0.05*c3
        h13 = 0.15*(s1*s3) - 0.05*c2
        h23 = 0.15*(s2*s3) + 0.05*c1

        # Stack into symmetric matrix (B,3,3)
        B = tf.stack([
            tf.concat([h11, h12, h13], axis=1),
            tf.concat([h12, h22, h23], axis=1),
            tf.concat([h13, h23, h33], axis=1)
        ], axis=1)  # (B,3,3) but with rows packed; fix to proper shape:
        # The above produced shape (3, B, 3); we want (B,3,3). So transpose:
        B = tf.transpose(B, perm=[1,0,2])

        # Ensure exact symmetry numerically and project to SPD
        S = 0.5*(B + tf.transpose(B, perm=[0,2,1])) + 0.0*tf.eye(3, dtype=tf.float64)[None, ...]
        g = make_spd_batch(S, min_eig=1e-2)  # (B,3,3) SPD
        return g

# ============================================================
# PINN (Laplace–Beltrami on 1-forms)
# ============================================================

class PINN:
    def __init__(self, metric_provider):
        self.metric_provider = metric_provider
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((3,), dtype=tf.float64),
            SineActivation(),
            tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(3, dtype=tf.float64)
        ])

    # ---------- Metric + Hodge stars ----------
    def metric_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return self.metric_provider.tensor(x)  # (B,3,3)

    def star_1form(self, alpha: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        alpha: (B,3)  →  returns 2-form in the usual [23,31,12] component order: (B,3)
        (*α)_{k} = sqrt|g| (g^{-1} α)_k  (with the standard index shuffle baked into the basis choice)
        Here we keep the same mapping you used: return components aligned with [dx2∧dx3, dx3∧dx1, dx1∧dx2].
        """
        g = self.metric_tensor(x)                 # (B,3,3)
        g_inv = tf.linalg.inv(g)                  # (B,3,3)
        sqrtg = tf.sqrt(tf.linalg.det(g))         # (B,)

        alpha_col = tf.expand_dims(alpha, -1)     # (B,3,1)
        v = tf.matmul(g_inv, alpha_col)           # (B,3,1)
        v = tf.squeeze(v, -1)                     # (B,3)
        return v * sqrtg[:, None]                 # (B,3)

    def star_2form(self, beta: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        beta: (B,3) with [b23, b31, b12]  →  returns 1-form (B,3)
        (*β)_k = 1/2 sqrt|g| ε_{kij} g^{ip} g^{jq} β_{pq}
        Implemented with batch einsums.
        """
        g = self.metric_tensor(x)                 # (B,3,3)
        g_inv = tf.linalg.inv(g)                  # (B,3,3)
        sqrtg = tf.sqrt(tf.linalg.det(g))         # (B,)

        b23, b31, b12 = tf.unstack(beta, axis=1)  # each (B,)
        zero = tf.zeros_like(b12)

        # Build full antisymmetric β_{ij} as (B,3,3)
        bmat = tf.stack([
            tf.stack([zero,  b12,  -b31], axis=1),
            tf.stack([-b12, zero,   b23], axis=1),
            tf.stack([b31, -b23,  zero], axis=1)
        ], axis=1)                                # (3,3,B) → fix to (B,3,3)
        bmat = tf.transpose(bmat, perm=[2,0,1])

        # term_{ij} = g_inv^{ip} g_inv^{jq} β_{pq}
        term = tf.einsum('bip,bjq,bpq->bij', g_inv, g_inv, bmat)  # (B,3,3)

        eps = tf.constant([
            [[0,0,0],[0,0,1],[0,-1,0]],
            [[0,0,-1],[0,0,0],[1,0,0]],
            [[0,1,0],[-1,0,0],[0,0,0]]
        ], dtype=tf.float64)  # (3,3,3)

        star_b = 0.5 * tf.einsum('bij,kij->bk', term, eps) * sqrtg[:, None]  # (B,3)
        return star_b

    def star_3form(self, w_scalar: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        w_scalar: (B,1) (coefficient of dx1∧dx2∧dx3) → returns 0-form (B,1)
        * (w vol) = w / sqrt|g|
        """
        g = self.metric_tensor(x)
        sqrtg = tf.sqrt(tf.linalg.det(g))         # (B,)
        return w_scalar / sqrtg[:, None]

    # ---------- Exterior derivatives ----------
    def grad_scalar(self, tape: tf.GradientTape, f: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        J = tape.batch_jacobian(f, x)             # (B,1,3)
        return tf.squeeze(J, axis=1)              # (B,3)

    def exterior_derivative_1_form(self, tape: tf.GradientTape, alpha: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        # alpha: (B,3) → returns (B,3) as [d23, d31, d12]
        J = tape.batch_jacobian(alpha, x)         # (B,3,3) J[:,i,μ] = ∂_μ α_i
        d23 = J[:, 2, 1] - J[:, 1, 2]
        d31 = J[:, 0, 2] - J[:, 2, 0]
        d12 = J[:, 1, 0] - J[:, 0, 1]
        return tf.stack([d23, d31, d12], axis=1)

    def exterior_derivative_2_form(self, tape: tf.GradientTape, beta: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        # beta: (B,3) [β23, β31, β12] → returns scalar 3-form (B,1): ∂1 β23 + ∂2 β31 + ∂3 β12
        J = tape.batch_jacobian(beta, x)          # (B,3,3) J[:,k,μ] = ∂_μ β_k
        w = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
        return tf.expand_dims(w, axis=1)          # (B,1)

    # ---------- PDE loss (Laplace–Beltrami on components) ----------
    def pde_error_at(self, x_point: tf.Tensor) -> tf.Tensor:
        x = tf.expand_dims(x_point, axis=0)       # (1,3)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.model(x)                     # (1,3)
            error = 0.0
            for i in range(3):
                f = u[:, i:i+1]                   # (1,1)
                df = self.grad_scalar(tape, f, x) # (1,3)
                star_df = self.star_1form(df, x)                  # (1,3) 2-form
                d_star_df = self.exterior_derivative_2_form(tape, star_df, x)  # (1,1) 3-form
                star_d_star_df = self.star_3form(d_star_df, x)    # (1,1) 0-form
                error += tf.reduce_sum(tf.square(star_d_star_df))
        del tape
        return error

    def loss(self, x_collocation: tf.Tensor) -> tf.Tensor:
        errs = tf.vectorized_map(self.pde_error_at, x_collocation)
        return tf.reduce_mean(errs)

    @tf.function
    def compute_loss_and_gradients(self, x_collocation: tf.Tensor):
        with tf.GradientTape() as tape:
            L = self.loss(x_collocation)
        grads = tape.gradient(L, self.model.trainable_variables)
        return L, grads

    def train(self, x_collocation: tf.Tensor, epochs: int = 200, lr: float = 1e-3, print_every: int = 10):
        opt = tf.keras.optimizers.Adam(lr)
        for ep in range(epochs):
            L, grads = self.compute_loss_and_gradients(x_collocation)
            opt.apply_gradients(zip(grads, self.model.trainable_variables))
            if ep % print_every == 0:
                print(f"Epoch {ep:4d} | Loss {L.numpy():.6e}")

    # ---------- Zero finder (uses g(x) at the evaluation point) ----------
    def find_zero_vector(self, tol: float = 1e-3):
        def objective(x_np):
            x_tensor = tf.convert_to_tensor(np.expand_dims(x_np, 0), dtype=tf.float64)   # (1,3)
            g = self.metric_tensor(x_tensor)[0]                                          # (3,3)
            u = self.evaluate(x_tensor)[0]                                              # (3,)
            u_col = tf.reshape(u, (3,1))
            val = tf.matmul(tf.transpose(u_col), tf.matmul(g, u_col))                    # (1,1)
            return float(tf.sqrt(val))

        result = differential_evolution(objective, bounds=[(0,1)]*3, polish=True)
        return result.fun < tol

    def evaluate(self, inputs: np.ndarray) -> tf.Tensor:
        return self.model(tf.convert_to_tensor(inputs, dtype=tf.float64))

# ============================================================
# Runner with random input-dependent metrics
# ============================================================

class PINNWithRandomInputMetrics(PINN):
    def __init__(self, seed: Optional[int] = None):
        self.metric_provider = InputDependentRandomMetric(seed)
        super().__init__(metric_provider=self.metric_provider)

    def reseed_metric(self, seed: Optional[int] = None):
        self.metric_provider.reseed(seed)

    def run_random_metrics(self,
                           x_collocation: tf.Tensor,
                           num_runs: int = 20,
                           train_epochs: int = 200,
                           lr: float = 1e-3,
                           zero_tol: float = 1e-3):
        # initial training on one random input-dependent metric
        print("Initial training on an input-dependent random metric…")
        self.train(x_collocation, epochs=train_epochs, lr=lr, print_every=10)

        zero_found_any = False
        for i in range(num_runs):
            print(f"\n--- Metric Run {i+1}/{num_runs} ---")
            self.reseed_metric()  # new spatial metric
            # brief fine-tune on the new metric (optional)
            self.train(x_collocation, epochs=50, lr=lr, print_every=50)
            found = self.find_zero_vector(tol=zero_tol)
            print("Zero 1-form found?" , found)
            zero_found_any = zero_found_any or found

        if not zero_found_any:
            print(f"No zeros found across {num_runs} input-dependent metrics.")
        return zero_found_any

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(0); tf.random.set_seed(0); random.seed(0)

    num_samples = 2000
    x_collocation = np.random.uniform(0, 1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    pinn = PINNWithRandomInputMetrics(seed=42)
    pinn.run_random_metrics(
        x_collocation,
        num_runs=20,          # bump to 100 when you’re happy with speed
        train_epochs=200,
        lr=1e-3,
        zero_tol=1e-3
    )
