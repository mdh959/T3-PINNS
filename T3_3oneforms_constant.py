import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Optional
from scipy.optimize import differential_evolution
import random

# ============================================================
# Utility functions and custom layers
# ============================================================

class SineActivation(Layer):
    """Simple sine–cosine activation layer to capture periodicity."""
    def call(self, inputs):
        return tf.concat([tf.sin(2 * np.pi * inputs), tf.cos(2 * np.pi * inputs)], axis=1)


def make_spd_from_symmetric(S: tf.Tensor) -> tf.Tensor:
    """Ensure S is symmetric positive definite by clipping eigenvalues."""
    w, v = tf.linalg.eigh(S)
    w = tf.clip_by_value(w, 1e-2, np.inf)
    return tf.matmul(v, tf.matmul(tf.linalg.diag(w), tf.transpose(v)))


def random_spd(seed: Optional[int] = None) -> tf.Tensor:
    """Generate a random 3x3 symmetric positive definite matrix."""
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
    A = tf.random.normal((3, 3), dtype=tf.float64)
    S = 0.5 * (A + tf.transpose(A))  # make symmetric
    return make_spd_from_symmetric(S)


# ============================================================
# Metric provider
# ============================================================

class ConstantRandomMetric:
    """Provides a single SPD metric g (constant in space)."""
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.g = random_spd(seed)

    def tensor(self, _: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self.g

    def reseed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.g = random_spd(seed)


# ============================================================
# Physics-Informed Neural Network (PINN)
# ============================================================

class PINN:
    """PINN solving Laplace-type equations for 1-forms on T3."""

    def __init__(self, metric_provider: ConstantRandomMetric):
        self.metric_provider = metric_provider
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((3,), dtype=tf.float64),
            SineActivation(),
            tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(3, dtype=tf.float64)
        ])

    # ------------------------------------------------------------
    # Metric and Hodge Star operators
    # ------------------------------------------------------------

    def metric_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return self.metric_provider.tensor(x)

    def star_1form(self, alpha: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Hodge star acting on a 1-form: returns components of a 2-form."""
        g = self.metric_tensor(x)
        g_inv = tf.linalg.inv(g)
        sqrtg = tf.sqrt(tf.linalg.det(g))

        alpha_col = tf.expand_dims(alpha, -1)
        v = tf.squeeze(tf.matmul(g_inv, alpha_col), -1)
        return sqrtg * v

    def star_2form(self, beta: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Hodge star acting on a 2-form beta = [b23, b31, b12]."""
        g = self.metric_tensor(x)
        g_inv = tf.linalg.inv(g)
        sqrtg = tf.sqrt(tf.linalg.det(g))

        b23, b31, b12 = tf.unstack(beta, axis=1)

        # assemble antisymmetric tensor b_ij
        bmat = tf.stack([
            tf.stack([tf.zeros_like(b12),  b12,       -b31], axis=1),
            tf.stack([-b12,                tf.zeros_like(b12), b23], axis=1),
            tf.stack([b31,                 -b23,      tf.zeros_like(b12)], axis=1)
        ], axis=1)

        # contraction g_inv^i_p g_inv^j_q b_pq
        term = tf.einsum('ip,jq,bpq->bij', g_inv, g_inv, bmat)

        eps = tf.constant([
            [[0, 0, 0], [0, 0, 1], [0,-1, 0]],
            [[0, 0,-1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [-1,0, 0], [0, 0, 0]]
        ], dtype=tf.float64)

        star_b = 0.5 * sqrtg * tf.einsum('kij,bij->bk', eps, term)
        return star_b
    

    def star_3form(self, w_scalar, x):
        g = self.metric_tensor(x)
        sqrtg = tf.sqrt(tf.linalg.det(g))
        # Common convention: *(w dx^1∧dx^2∧dx^3) = w / sqrt(|g|)
        return w_scalar / sqrtg

    # ------------------------------------------------------------
    # Differential operators
    # ------------------------------------------------------------

    def grad_scalar(self, tape: tf.GradientTape, f: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        J = tape.batch_jacobian(f, x)
        return tf.squeeze(J, axis=1)

    def exterior_derivative_1_form(self, tape: tf.GradientTape, alpha: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        J = tape.batch_jacobian(alpha, x)
        d23 = J[:, 2, 1] - J[:, 1, 2]
        d31 = J[:, 0, 2] - J[:, 2, 0]
        d12 = J[:, 1, 0] - J[:, 0, 1]
        return tf.stack([d23, d31, d12], axis=1)
    
    def exterior_derivative_2_form(self, tape, beta, x):
        """
        Exterior derivative of a 2-form beta = [beta23, beta31, beta12].
        Returns a 3-form (scalar coefficient of dx1∧dx2∧dx3).
        """
        J = tape.batch_jacobian(beta, x)  # shape (B, 3, 3); J[:,k,μ] = ∂_μ β_k
        # (dbeta)_{123} = ∂1 beta23 + ∂2 beta31 + ∂3 beta12
        w = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
        return tf.expand_dims(w, axis=1)


    # ------------------------------------------------------------
    # PDE Loss
    # ------------------------------------------------------------

    def pde_error(self, x):
        # x shape: (3,)
        x = tf.expand_dims(x, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.model(x)  # (1,3), interpreted as a 1-form

            # d u: (1,3) -> (1,3) in [23,31,12] components
            d_u = self.exterior_derivative_1_form(tape, u, x)

            # dela_u = * d (*u)
            star_u = self.star_1form(u, x)                           # 2-form (1,3)
            d_star_u = self.exterior_derivative_2_form(tape, star_u, x)  # 3-form (1,1)
            delta_u = self.star_3form(d_star_u, x)                   # 0-form (1,1)

            # Total error: ||d u||² + ||delta_u||²
            err_d = tf.reduce_sum(tf.square(d_u))
            err_delta = tf.reduce_sum(tf.square(delta_u))
            error_total = err_d + err_delta

        return error_total


    def loss(self, x_collocation: tf.Tensor) -> tf.Tensor:
        errs = tf.vectorized_map(self.pde_error, x_collocation)
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

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def test_hodge_involution(self) -> None:
        """Check numerically that *(*alpha)=alpha and *(*beta)=beta."""
        x = tf.zeros((1,3), dtype=tf.float64)
        a = tf.random.normal((1,3), dtype=tf.float64)
        b = tf.random.normal((1,3), dtype=tf.float64)

        err1 = tf.norm(self.star_2form(self.star_1form(a, x), x) - a).numpy()
        err2 = tf.norm(self.star_1form(self.star_2form(b, x), x) - b).numpy()
        print(f"Hodge check: ||*(*a)-a||={err1:.3e}, ||*(*b)-b||={err2:.3e}")

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def evaluate(self, inputs: np.ndarray) -> tf.Tensor:
        return self.model(tf.convert_to_tensor(inputs, dtype=tf.float64))


# ============================================================
# PINN variant with random metrics
# ============================================================

class PINNWithRandomMetrics(PINN):
    """Runs the PINN across multiple random constant metrics."""
    def __init__(self, seed: Optional[int] = None):
        self.metric_provider = ConstantRandomMetric(seed)
        super().__init__(metric_provider=self.metric_provider)

    def reseed_metric(self, seed: Optional[int] = None) -> None:
        self.metric_provider.reseed(seed)

    def find_zero_vector(self, tol: float = 1e-3):
        """Minimize norm of the 1-form u(x) using global optimization."""
        g = self.metric_tensor(None)

        def objective(x_np):
            x_tensor = tf.convert_to_tensor(np.expand_dims(x_np, 0), dtype=tf.float64)
            u = self.evaluate(x_tensor)[0]
            u_col = tf.reshape(u, (3,1))
            val = tf.matmul(tf.transpose(u_col), tf.matmul(g, u_col))
            return float(tf.sqrt(val))

        result = differential_evolution(objective, bounds=[(-np.pi, np.pi)]*3, polish=True)
        if result.fun < tol:
            return True, g
        return False, None

    def run_random_metrics(self,
                           x_collocation: tf.Tensor,
                           num_runs: int = 100,
                           train_epochs: int = 200,
                           lr: float = 1e-3,
                           zero_tol: float = 1e-3):
        """Train once, then test on 100 random constant metrics."""
        print("Initial training on random metric...")
        self.train(x_collocation, epochs=train_epochs, lr=lr, print_every=10)
        self.test_hodge_involution()

        zero_found = False
        for i in range(num_runs):
            print(f"\n--- Metric Run {i+1}/{num_runs} ---")
            self.reseed_metric()
            self.train(x_collocation, epochs=30, lr=lr, print_every=30)
            found, g = self.find_zero_vector(tol=zero_tol)
            if found:
                zero_found = True
                print("Zero 1-form found for metric:\n", g.numpy())

        if not zero_found:
            print(f"No zero 1-forms found after {num_runs} random metrics.")

        return zero_found


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)

    num_samples = 1500
    x_collocation = np.random.uniform(0, 1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    pinn = PINNWithRandomMetrics(seed=42)
    pinn.run_random_metrics(
        x_collocation,
        num_runs=100,
        train_epochs=200,
        lr=1e-3,
        zero_tol=1e-3
    )
