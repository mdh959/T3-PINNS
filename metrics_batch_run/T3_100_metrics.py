import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from scipy.linalg import expm  # Matrix exponential
from scipy.optimize import differential_evolution
import random

class SineActivation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2 * np.pi * inputs), tf.cos(2 * np.pi * inputs)], 1)


class RandomMetricGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
    
    def generate_metric_tensor(self):
        # Generate a random 3x3 matrix A
        A = tf.random.normal((3, 3), dtype=tf.float64)
        # Make it symmetric: M = (A + A^T) / 2
        symmetric_matrix = (A + tf.transpose(A)) / 2
        return tf.linalg.expm(symmetric_matrix)
    
class PINN:
    def __init__(self):
        # define the neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,), dtype=tf.float64),  # Input layer with float64
            SineActivation(),
            tf.keras.layers.Dense(units=32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(units=64, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(units=32, activation='tanh', dtype=tf.float64),
            tf.keras.layers.Dense(units=3, dtype=tf.float64)  # Output layer for f1, f2, f3
        ])

    def partial_derivative(self, tape, u, x, dim):
        du_dx = tape.gradient(u, x)
        return du_dx[:, dim]  # extracting the partial derivative w.r.t specified dimension

    def hodge_star(self, u, x):
        u1, u2, u3 = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    
        # Define the metric tensor g and compute its determinant and inverse
        g = self.metric_tensor(x)
        g_det = tf.linalg.det(g)
        g_inv = tf.linalg.inv(g)
    
        # Construct the (u1, u2, u3) vector
        u_vec = tf.concat([u1, -u2, u3], axis=1)
        sqrt_det_g = tf.sqrt(g_det)
        sqrt_det_g = tf.reshape(sqrt_det_g, (1, 1))  # Ensure the shape matches for broadcasting
        u_vec = tf.expand_dims(u_vec, axis=1)  # Add an extra dimension for matrix multiplication
        # Compute the sqrt(det(g)) * g_inv * u_vec
        transformed_vec = sqrt_det_g * tf.matmul(u_vec, g_inv)
        transformed_vec = tf.squeeze(transformed_vec, axis=1)
    
        # Extract the components
        alpha = transformed_vec[:, 0:1]
        beta = transformed_vec[:, 1:2]
        gamma = transformed_vec[:, 2:3]
    
        # Hodge star operation
        star_u = tf.concat([
            alpha,    # coefficient of dx2 ^ dx3
            beta,    # coefficient of dx3 ^ dx1
            gamma     # coefficient of dx1 ^ dx2
        ], axis=1)
    
        return star_u

    def exterior_derivative_1_form(self, tape, u, x): # Exterior derivative on a 1-form
        du1_dx2 = self.partial_derivative(tape, u[:, 0], x, 1)  # Partial derivative with respect to x2
        du1_dx3 = self.partial_derivative(tape, u[:, 0], x, 2)  # Partial derivative with respect to x3
        du2_dx1 = self.partial_derivative(tape, u[:, 1], x, 0)  # Partial derivative with respect to x1
        du2_dx3 = self.partial_derivative(tape, u[:, 1], x, 2)  # Partial derivative with respect to x3
        du3_dx1 = self.partial_derivative(tape, u[:, 2], x, 0)  # Partial derivative with respect to x1
        du3_dx2 = self.partial_derivative(tape, u[:, 2], x, 1)  # Partial derivative with respect to x2

        d_u = tf.stack([du2_dx3 -du3_dx2,du3_dx1 - du1_dx3, du1_dx2 -du2_dx1], axis=1)

        return d_u # Output is a 2-form

    def star_derivative_2_form(self, tape, u, x): # Hodge star on 3-form
        du1_dx1 = self.partial_derivative(tape, u[:, 0], x, 0)
        du2_dx2 = self.partial_derivative(tape, u[:, 1], x, 1)
        du3_dx3 = self.partial_derivative(tape, u[:, 2], x, 2)
        g_det = tf.linalg.det(self.metric_tensor(x))
        # Compute the divergence of the 1-form
        divergence = du1_dx1 + du2_dx2 + du3_dx3
        sqrt_det_g = tf.sqrt(g_det)
        # Apply the Hodge star operation (multiplying by sqrt(det(g)))
        star_divergence = divergence * sqrt_det_g
        return star_divergence # Output is a 0-form

    def derivative_function(self, tape, u, x): # Exterior derivative on a 0-form
        df_dx1 = self.partial_derivative(tape, u, x, 0)  # Partial derivative with respect to x1
        df_dx2 = self.partial_derivative(tape, u, x, 1)  # Partial derivative with respect to x2
        df_dx3 = self.partial_derivative(tape, u, x, 2)  # Partial derivative with respect to x3
        return tf.stack([df_dx1, df_dx2, df_dx3], axis=1) # Output is a 1-form
    
    def pde_error(self, x, tape):
        error = 0
        x = tf.expand_dims(x, axis=0)
        tape.watch(x)
        u = self.model(x)
    
        # Extract f1, f2, f3 from u
        f1, f2, f3 = tf.expand_dims(u[0, 0], axis=0), tf.expand_dims(u[0, 1], axis=0), tf.expand_dims(u[0, 2], axis=0)
        f1, f2, f3 = tf.expand_dims(f1, axis=-1), tf.expand_dims(f2, axis=-1), tf.expand_dims(f3, axis=-1)

        # Loop over f1, f2, f3
        for i, f in enumerate([f1, f2, f3]):
            # Calculate exterior derivative of f
            d_f = self.derivative_function(tape, f, x)

            # Modify specific element of df based on index i
            if i == 0:  # For f1, add 1 to the first element
                d_f = tf.concat([d_f[:, 0:1] + 1, d_f[:, 1:2], d_f[:, 2:3]], axis=1)
            elif i == 1:  # For f2, add 1 to the second element
                d_f = tf.concat([d_f[:, 0:1], d_f[:, 1:2] + 1, d_f[:, 2:3]], axis=1)
            elif i == 2:  # For f3, add 1 to the third element
                d_f = tf.concat([d_f[:, 0:1], d_f[:, 1:2], d_f[:, 2:3] + 1], axis=1)

            # Apply Hodge star, exterior derivative, and Hodge star again
            hodge_star_d_f = self.hodge_star(d_f, x)
            d_hodge_star_d_f = self.exterior_derivative_1_form(tape, hodge_star_d_f, x)
            hodge_star_d_hodge_star_d_f = self.hodge_star(d_hodge_star_d_f, x)
            error = tf.reduce_sum(tf.square(hodge_star_d_hodge_star_d_f))
    
        return error

    def loss(self, x_collocation, tape):
        # Compute the PDE error for each collocation point
        errors = tf.vectorized_map(lambda x: self.pde_error(x, tape), x_collocation)
        return tf.reduce_mean(errors)

    @tf.function
    def compute_loss_and_gradients(self, x_collocation):
        with tf.GradientTape(persistent=True) as tape:
            normalised_loss = self.loss(x_collocation, tape)
        grads = tape.gradient(normalised_loss, self.model.trainable_variables)
        return normalised_loss, grads

    def train(self, x_collocation, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            normalised_loss, grads = self.compute_loss_and_gradients(x_collocation)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {normalised_loss.numpy()}")
    
    def evaluate(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float64)
        outputs = self.model(inputs)
        return outputs

class PINNWithRandomMetrics(PINN):
    def __init__(self):
        super().__init__()
        # Create the random metric generator once, so the same one can be used for all metrics
        self.random_metric_generator = RandomMetricGenerator()

    def metric_tensor(self, x):
        # Use the same random metric generator for each call
        random_metric = self.random_metric_generator.generate_metric_tensor()
        g = tf.convert_to_tensor(random_metric, dtype=tf.float64)
        return g
    
    def run_random_metrics(self, x_collocation, num_runs=100):
        # Train the model once before running the different metrics
        self.train(x_collocation, epochs=100, learning_rate=0.001)  # Train the model once

        zero_1_form_found = False
        for i in range(num_runs):
            print(f"Running iteration {i+1}/{num_runs}")
            # Generate a new random metric for each run
            self.metric_tensor = self.metric_tensor  # Ensures new metric each iteration
        
            zero_found, metric_at_zero = self.find_zero_vector()
            if zero_found:
                zero_1_form_found = True
                print(f"Zero 1-form found at iteration {i+1}")
                print(f"Metric tensor at the zero vector: {metric_at_zero}")
                
    
        if not zero_1_form_found:
            print(f"No zero 1-forms found after {num_runs} runs.")
    
        return zero_1_form_found

    def find_zero_vector(self):
        def objective_function(x):
            x_tensor = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float64)
            g = self.metric_tensor(x_tensor)
            u = self.evaluate(x_tensor)[0]
            u = tf.reshape(u, (3, 1))
            
            # Compute the metric norm: sqrt(u^T * g * u)
            metric_norm_squared = tf.matmul(tf.matmul(tf.transpose(u), g), u)
            metric_norm = tf.sqrt(metric_norm_squared)
            return tf.squeeze(metric_norm).numpy()

        # Perform a global optimization using Differential Evolution
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        result = differential_evolution(objective_function, bounds)
        min_norm_final = result.fun
        min_point_final = result.x
        min_point_tensor = tf.convert_to_tensor(np.expand_dims(min_point_final, axis=0), dtype=tf.float64)
        metric_at_zero = self.metric_tensor(min_point_tensor)

        # Check if the smallest norm is sufficiently close to zero
        if min_norm_final < 1e-4:
            print(f"Zero vector found: {min_norm_final} at point {min_point_final}")
            return True, metric_at_zero 
        else:
            print(f"No zero vector found, smallest norm: {min_norm_final}")
            return False, None

# Example usage
if __name__ == '__main__':
    num_samples = 1000
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    # Initialize PINN model with random metrics
    pinn = PINNWithRandomMetrics()

    # Run with random metrics multiple times, using the same trained parameters
    pinn.run_random_metrics(x_collocation, num_runs=100)
