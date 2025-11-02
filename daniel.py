import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from scipy.optimize import differential_evolution
import random

tf.keras.backend.set_floatx('float64')


# ============================================================
# Periodic activation for T^3
# ============================================================
class TrigActivation(Layer):
    def call(self, inputs):
        return tf.concat([tf.sin(2 * np.pi * inputs), tf.cos(2 * np.pi * inputs)], axis=1)


# ============================================================
# Utility: random symmetric positive definite metric
# ============================================================
def make_spd_from_symmetric(S):
    w, v = tf.linalg.eigh(S)
    w = tf.clip_by_value(w, 1e-2, np.inf)
    return tf.matmul(v, tf.matmul(tf.linalg.diag(w), tf.transpose(v)))


def random_spd(seed=None):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
    A = tf.random.normal((3, 3), dtype=tf.float64)
    S = 0.5 * (A + tf.transpose(A))
    return make_spd_from_symmetric(S)


class ConstantRandomMetric:
    """A constant SPD metric independent of x."""
    def __init__(self, seed=None):
        self.g = random_spd(seed)

    def tensor(self, x=None):
        return self.g

    def reseed(self, seed=None):
        self.g = random_spd(seed)


# ============================================================
# PINN for harmonic 1-forms on T^3
# ============================================================
class PINN:
    def __init__(self, metric_provider):
        self.metric_provider = metric_provider

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((3,), dtype=tf.float64),
            TrigActivation(),
            tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(3, dtype=tf.float64)
        ])

    # ------------------------------------------------------------
    # Metric and Hodge star operators
    # ------------------------------------------------------------
    def metric_tensor(self, x):
        return self.metric_provider.tensor(x)

    def star_1form(self, alpha, x):
        """Hodge star on a 1-form (returns a 2-form)."""
        g = self.metric_tensor(x)
        g_inv = tf.linalg.inv(g)
        sqrt_g = tf.sqrt(tf.linalg.det(g))
        alpha_col = tf.expand_dims(alpha, -1)
        v = tf.squeeze(tf.matmul(g_inv, alpha_col), -1)
        return sqrt_g * v

    def star_3form(self, omega, x):
        """Hodge star on a 3-form (returns a scalar)."""
        g = self.metric_tensor(x)
        sqrt_g = tf.sqrt(tf.linalg.det(g))
        return omega / sqrt_g

    # ------------------------------------------------------------
    # Exterior derivatives
    # ------------------------------------------------------------
    def exterior_derivative_1_form(self, tape, alpha, x):
        """d(alpha) for a 1-form alpha -> 2-form."""
        J = tape.batch_jacobian(alpha, x)
        d23 = J[:, 2, 1] - J[:, 1, 2]
        d31 = J[:, 0, 2] - J[:, 2, 0]
        d12 = J[:, 1, 0] - J[:, 0, 1]
        return tf.stack([d23, d31, d12], axis=1)

    def exterior_derivative_2_form(self, tape, beta, x):
        """d(beta) for a 2-form beta -> 3-form (scalar)."""
        J = tape.batch_jacobian(beta, x)
        w = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
        return tf.expand_dims(w, axis=1)

    # ------------------------------------------------------------
    # Metric-weighted norm of a 2-form
    # ------------------------------------------------------------
    def metric_norm_of_2form(self, beta, x):
        """Compute the squared norm ||beta||^2_g."""
        g = self.metric_tensor(x)
        g_inv = tf.linalg.inv(g)

        b23, b31, b12 = tf.unstack(beta, axis=1)
        beta_mat = tf.stack([
            tf.stack([tf.zeros_like(b12), b12, -b31], axis=1),
            tf.stack([-b12, tf.zeros_like(b12), b23], axis=1),
            tf.stack([b31, -b23, tf.zeros_like(b12)], axis=1)
        ], axis=1)

        term = tf.einsum('ip,jq,bij,bpq->b', g_inv, g_inv, beta_mat, beta_mat)
        return term  # squared norm

    # ------------------------------------------------------------
    # PDE error: harmonic 1-form condition
    # ------------------------------------------------------------
    def pde_error_at(self, x_point):
        x = tf.expand_dims(x_point, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.model(x)

            # Compute dλ (2-form)
            d_lambda = self.exterior_derivative_1_form(tape, u, x)

            # Compute *d(*λ) (0-form)
            star_lambda = self.star_1form(u, x)
            d_star_lambda = self.exterior_derivative_2_form(tape, star_lambda, x)
            star_d_star_lambda = self.star_3form(d_star_lambda, x)

            # Combine both loss components
            norm_d_lambda_sq = self.metric_norm_of_2form(d_lambda, x)
            norm_star_d_star_lambda_sq = tf.square(star_d_star_lambda)

            total_error = norm_d_lambda_sq + norm_star_d_star_lambda_sq

        del tape
        return total_error

    # ------------------------------------------------------------
    # Loss and training
    # ------------------------------------------------------------
    def loss(self, x_collocation):
        errors = tf.vectorized_map(self.pde_error_at, x_collocation)
        return tf.reduce_mean(errors)

    @tf.function
    def compute_loss_and_gradients(self, x_collocation):
        with tf.GradientTape() as tape:
            L = self.loss(x_collocation)
        grads = tape.gradient(L, self.model.trainable_variables)
        return L, grads

    def train(self, x_collocation, epochs=200, lr=1e-3, print_every=10):
        opt = tf.keras.optimizers.Adam(lr)
        for ep in range(epochs):
            L, grads = self.compute_loss_and_gradients(x_collocation)
            opt.apply_gradients(zip(grads, self.model.trainable_variables))
            if ep % print_every == 0:
                print(f"Epoch {ep:4d} | Loss {L.numpy():.6e}")

    # ------------------------------------------------------------
    # Zero finder for the trained 1-form
    # ------------------------------------------------------------
    def find_zero_vector(self, tol=1e-3):
        g = self.metric_tensor(None)

        def objective(x_np):
            x_tensor = tf.convert_to_tensor(np.expand_dims(x_np, 0), dtype=tf.float64)
            u = self.model(x_tensor)[0]
            u_col = tf.reshape(u, (3, 1))
            val = tf.matmul(tf.transpose(u_col), tf.matmul(g, u_col))
            return float(tf.sqrt(val))

        result = differential_evolution(objective, bounds=[(0, 1)] * 3, polish=True)
        if result.fun < tol:
            print(f"Zero 1-form found at {result.x}, |u|={result.fun:.3e}")
            return True, g
        print(f"No zero found, min norm = {result.fun:.3e}")
        return False, None


# ============================================================
# Wrapper: run across multiple random metrics
# ============================================================
class PINNWithRandomMetrics(PINN):
    def __init__(self, seed=None):
        self.metric_provider = ConstantRandomMetric(seed)
        super().__init__(metric_provider=self.metric_provider)

    def reseed_metric(self, seed=None):
        self.metric_provider.reseed(seed)

    def run_random_metrics(self, x_collocation, num_runs=100, train_epochs=200, lr=1e-3, zero_tol=1e-3):
        print("Initial training on random metric...")
        self.train(x_collocation, epochs=train_epochs, lr=lr, print_every=10)

        zero_found = False
        for i in range(num_runs):
            print(f"\n--- Metric Run {i+1}/{num_runs} ---")
            self.reseed_metric()
            self.train(x_collocation, epochs=30, lr=lr, print_every=30)
            found, g = self.find_zero_vector(tol=zero_tol)
            if found:
                zero_found = True
                print("Metric yielding zero 1-form:\n", g.numpy())

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
