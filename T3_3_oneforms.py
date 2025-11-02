# pinn_T3_random_metrics.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from scipy.optimize import differential_evolution
import time

# ---------- Activation ----------
class SineActivation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float64)
        return tf.concat([tf.sin(2.0 * np.pi * inputs), tf.cos(2.0 * np.pi * inputs)], axis=1)

# ---------- PINN ----------
class PINN:
    def __init__(self, metric_generator=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,), dtype=tf.float64),
            SineActivation(),
            tf.keras.layers.Dense(units=32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(units=64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(units=32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(units=3, dtype=tf.float64)
        ])

        self.metric_generator = metric_generator
        self.current_metric = None

    def set_current_metric(self, metric_tensor):
        self.current_metric = tf.cast(metric_tensor, tf.float64)

    def metric_tensor(self, x=None):
        # If current metric not set, generate a new one (fixed metric)
        if self.current_metric is None:
            if self.metric_generator is not None:
                self.set_current_metric(self.metric_generator.generate_metric_tensor())
            else:
                # Default metric: identity
                self.set_current_metric(tf.eye(3, dtype=tf.float64))
        return self.current_metric

    def partial_derivative(self, tape, u, x, dim):
        du_dx = tape.gradient(u, x)
        return du_dx[:, dim]

    def hodge_star(self, u, x):
        g = self.metric_tensor(x)
        g_det = tf.linalg.det(g)
        g_inv = tf.linalg.inv(g)

        # u shape: (batch, 3)
        u_vec = tf.expand_dims(tf.concat([u[:, 0:1], -u[:, 1:2], u[:, 2:3]], axis=1), axis=2)
        sqrt_det_g = tf.sqrt(g_det)

        g_inv_batch = tf.expand_dims(g_inv, axis=0)
        transformed = sqrt_det_g * tf.matmul(g_inv_batch, u_vec)
        transformed = tf.squeeze(transformed, axis=2)
        return transformed

    def exterior_derivative_1_form(self, tape, u, x):
        du1_dx2 = self.partial_derivative(tape, u[:, 0], x, 1)
        du1_dx3 = self.partial_derivative(tape, u[:, 0], x, 2)
        du2_dx1 = self.partial_derivative(tape, u[:, 1], x, 0)
        du2_dx3 = self.partial_derivative(tape, u[:, 1], x, 2)
        du3_dx1 = self.partial_derivative(tape, u[:, 2], x, 0)
        du3_dx2 = self.partial_derivative(tape, u[:, 2], x, 1)
        d_u = tf.stack([du2_dx3 - du3_dx2,
                        du3_dx1 - du1_dx3,
                        du1_dx2 - du2_dx1], axis=1)
        return d_u

    def derivative_function(self, tape, u, x):
        df_dx1 = self.partial_derivative(tape, u, x, 0)
        df_dx2 = self.partial_derivative(tape, u, x, 1)
        df_dx3 = self.partial_derivative(tape, u, x, 2)
        return tf.stack([df_dx1, df_dx2, df_dx3], axis=1)

    def pde_error(self, x):
        # x shape: (3,)
        x = tf.expand_dims(x, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = self.model(x)  # (1,3)

            error_total = 0.0
            for i in range(3):
                f = tf.expand_dims(u[:, i], axis=-1)  # shape (1,1)
                d_f = self.derivative_function(tape, f, x)

                # Add 1 to component i
                d_f = tf.concat([
                    d_f[:, 0:1] + (1.0 if i == 0 else 0.0),
                    d_f[:, 1:2] + (1.0 if i == 1 else 0.0),
                    d_f[:, 2:3] + (1.0 if i == 2 else 0.0)
                ], axis=1)

                hodge_star_d_f = self.hodge_star(d_f, x)
                d_hodge_star_d_f = self.exterior_derivative_1_form(tape, hodge_star_d_f, x)
                hodge_star_d_hodge_star_d_f = self.hodge_star(d_hodge_star_d_f, x)
                error_total += tf.reduce_sum(tf.square(hodge_star_d_hodge_star_d_f))

        return error_total

    def loss(self, x_collocation):
        # Vectorize PDE error computation over batch using tf.vectorized_map
        errors = tf.vectorized_map(self.pde_error, x_collocation)
        return tf.reduce_mean(errors)

    @tf.function
    def compute_loss_and_gradients(self, x_collocation):
        with tf.GradientTape(persistent=True) as tape:
            loss_value = self.loss(x_collocation)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        return loss_value, grads

    def train(self, x_collocation, epochs=100, learning_rate=1e-3, verbose=True):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            loss_value, grads = self.compute_loss_and_gradients(x_collocation)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Loss = {loss_value.numpy():.6e}")

    def evaluate(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
        return self.model(inputs)

    def find_zero_vector(self, bounds=[(-np.pi, np.pi)] * 3):
        if self.current_metric is None:
            raise ValueError("Current metric not set. Call set_current_metric() before find_zero_vector()")

        metric_np = self.current_metric.numpy()

        def objective(x_np):
            x_tf = tf.convert_to_tensor(np.expand_dims(x_np, axis=0), dtype=tf.float64)
            u = self.model(x_tf)[0].numpy().reshape((3, 1))
            val = float(np.sqrt((u.T @ metric_np @ u).squeeze()))
            return val

        result = differential_evolution(objective, bounds)
        return result.fun, result.x

    def run_random_metrics_train_per_metric(self, x_collocation, num_runs=100, epochs_per_metric=100,
                                           lr=1e-3, zero_threshold=1e-3, reinit_weights=True):
        results = []
        start_time = time.time()

        for i in range(num_runs):
            print(f"\n=== Run {i + 1}/{num_runs} ===")
            if self.metric_generator is not None:
                metric = self.metric_generator.generate_metric_tensor()
            else:
                metric = tf.eye(3, dtype=tf.float64)

            self.set_current_metric(metric)

            if reinit_weights:
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
                        kernel_init = layer.kernel_initializer
                        bias_init = layer.bias_initializer
                        if hasattr(layer, 'kernel'):
                            layer.kernel.assign(kernel_init(tf.shape(layer.kernel)))
                        if hasattr(layer, 'bias'):
                            layer.bias.assign(bias_init(tf.shape(layer.bias)))

            self.train(x_collocation, epochs=epochs_per_metric, learning_rate=lr, verbose=False)

            min_norm, min_point = self.find_zero_vector()

            print(f"Trained; final min norm = {min_norm:.6e} at point {min_point}")

            results.append({
                'index': i + 1,
                'min_norm': min_norm,
                'min_point': min_point,
                'metric': metric.numpy()
            })

        elapsed = time.time() - start_time
        print(f"\nCompleted {num_runs} runs in {elapsed:.1f} seconds.")

        found = [r for r in results if r['min_norm'] < zero_threshold]
        print(f"\nFound {len(found)} metrics with min norm < {zero_threshold}")

        for r in found:
            print(f"Run {r['index']}: min_norm = {r['min_norm']:.6e}, min_point = {r['min_point']}")
            print("Metric tensor:")
            print(r['metric'])
            print("-" * 40)

        return results


# Example usage
if __name__ == '__main__':
    num_samples = 500
    x_collocation = np.random.uniform(0, 1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    pinn = PINN()
    pinn.run_random_metrics_train_per_metric(x_collocation, num_runs=10, epochs_per_metric=50)
