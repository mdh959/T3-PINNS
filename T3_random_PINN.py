import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Set the floating point precision
tf.keras.backend.set_floatx('float64')

class SineActivation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2 * np.pi * inputs), tf.cos(2 * np.pi * inputs)], 1)

class PINN:
    def __init__(self):
        # define the neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,)),  # Input layer with 3 inputs: x1, x2, x3
            SineActivation(),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=64, activation='tanh'),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=3)  # Output layer for f1, f2, f3
        ])
    def metric_tensor(self,x):
        x=x[0]
        g11 = 1.0 
        g12 = 0.0
        g13 = 0.0
        g22 = 1.9*(x[0]-0.5)**2
        g23 = 0.0
        g33 = 1.9*(x[1]-0.5)**2
        # could instead multiply identity by conformal_factor = 1.01 + tf.sin(2 * np.pi * x[0]) * \
                               #tf.sin(2 * np.pi * x[1]) * \
                               #tf.sin(2 * np.pi * x[2])
    
        g = tf.convert_to_tensor([
            [g11, g12, g13],
            [g12, g22, g23],
            [g13, g23, g33]
        ], dtype=tf.float64)
        
        return g

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

    def exterior_derivative_1_form(self, tape, u, x):
        du1_dx2 = self.partial_derivative(tape, u[:, 0], x, 1)  # Partial derivative with respect to x2
        du1_dx3 = self.partial_derivative(tape, u[:, 0], x, 2)  # Partial derivative with respect to x3
        du2_dx1 = self.partial_derivative(tape, u[:, 1], x, 0)  # Partial derivative with respect to x1
        du2_dx3 = self.partial_derivative(tape, u[:, 1], x, 2)  # Partial derivative with respect to x3
        du3_dx1 = self.partial_derivative(tape, u[:, 2], x, 0)  # Partial derivative with respect to x1
        du3_dx2 = self.partial_derivative(tape, u[:, 2], x, 1)  # Partial derivative with respect to x2

        d_u = tf.stack([du2_dx3 -du3_dx2,du3_dx1 - du1_dx3, du1_dx2 -du2_dx1], axis=1)

        return d_u

    def star_derivative_2_form(self, tape, u, x):
        du1_dx1 = self.partial_derivative(tape, u[:, 0], x, 0)
        du2_dx2 = self.partial_derivative(tape, u[:, 1], x, 1)
        du3_dx3 = self.partial_derivative(tape, u[:, 2], x, 2)
        g_det = tf.linalg.det(self.metric_tensor(x))
        # Compute the divergence of the 1-form
        divergence = du1_dx1 + du2_dx2 + du3_dx3
        sqrt_det_g = tf.sqrt(g_det)
        # Apply the Hodge star operation (multiplying by sqrt(det(g)))
        star_divergence = divergence * sqrt_det_g
        return star_divergence

    def derivative_function(self, tape, u, x):
        df_dx1 = self.partial_derivative(tape, u, x, 0)  # Partial derivative with respect to x1
        df_dx2 = self.partial_derivative(tape, u, x, 1)  # Partial derivative with respect to x2
        df_dx3 = self.partial_derivative(tape, u, x, 2)  # Partial derivative with respect to x3
        return tf.stack([df_dx1, df_dx2, df_dx3], axis=1)
    
    def pde_error(self, x, tape):
        x = tf.expand_dims(x, axis=0)
        tape.watch(x)
        u = self.model(x)
        hodge_star_u = self.hodge_star(u, x)
        hodge_star_d_hodge_star_u = self.star_derivative_2_form(tape, hodge_star_u, x)
        d_hodge_star_d_hodge_star_u = self.derivative_function(tape, hodge_star_d_hodge_star_u, x)

        d_u = self.exterior_derivative_1_form(tape, u, x)
        hodge_star_d_u = self.hodge_star(d_u, x)
        d_hodge_star_d_u = self.exterior_derivative_1_form(tape, hodge_star_d_u, x)
        hodge_star_d_hodge_star_d_u = self.hodge_star(d_hodge_star_d_u, x)

        sum_tensor = hodge_star_d_hodge_star_d_u + d_hodge_star_d_hodge_star_u
        error = tf.reduce_sum(tf.square(sum_tensor))
        return error

    def loss(self, x_collocation, tape):
        # Compute the PDE error for each collocation point
        errors = tf.vectorized_map(lambda x: self.pde_error(x, tape), x_collocation)
        norm_factor = tf.reduce_sum(tf.abs(self.model(x_collocation)))
        normalised_loss = tf.reduce_mean(errors) / norm_factor
        return normalised_loss

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

    def plot_learned_1_form(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Make the grid
        x, y, z = np.meshgrid(np.arange(0, 1.1, 0.2),
                              np.arange(0, 1.1, 0.2),
                              np.arange(0, 1.1, 0.2))
        grid_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        grid_tensor = tf.convert_to_tensor(grid_points, dtype=tf.float64)

        # Evaluate the model
        u = self.evaluate(grid_tensor).numpy()

        # Reshape to match grid
        u1 = u[:, 0].reshape(x.shape)
        u2 = u[:, 1].reshape(x.shape)
        u3 = u[:, 2].reshape(x.shape)
        ax.quiver(x, y, z, u1, u2, u3, length=0.1, normalize=True)

        plt.show()
    
    def find_zero_vector(self, num_points_list=[100, 1000, 10000, 100000, 1e6]):
        for num_points in num_points_list:
            print(f"\nTesting with {int(num_points)} points.")

            # Generate random points within the unit cube
            random_points = np.random.uniform(low=0, high=1, size=(int(num_points), 3))
            random_points_tensor = tf.convert_to_tensor(random_points, dtype=tf.float64)

            # Function to calculate the metric norm
            def calculate_metric_norm(x):
                x = tf.expand_dims(x, axis=0)  # Adjust shape to match the metric function input
                g = self.metric_tensor(x)  # Get the metric tensor for the single point

                # Evaluate the model on these points
                u = self.evaluate(x)[0]  # Get the output vector (1-form)

                # Reshape u to be a column vector
                u = tf.reshape(u, (3, 1))

                # Calculate the metric norm using g_ij u^i u^j
                metric_norm_squared = tf.matmul(tf.matmul(tf.transpose(u), g), u)
                metric_norm = tf.sqrt(metric_norm_squared)

                # Return a scalar
                return tf.squeeze(metric_norm)

            # Calculate norms using vectorized_map for speed
            norms = tf.vectorized_map(calculate_metric_norm, random_points_tensor).numpy()

            # Find the index of the minimum norm
            min_index = np.argmin(norms)
            min_norm = norms[min_index]
            min_point = random_points[min_index]

            # Print the result of the random search
            if min_norm < 1e-3:
                print(f"Found a point with small norm at {min_point} with norm {min_norm} during random search.")
            else:
                print("No point with sufficiently small norm found during random search.")

            print("Smallest norm found during random search: " + str(min_norm))

            # Define the objective function for minimization
            def objective_function(x):
                x_tensor = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float64)
                g = self.metric_tensor(x_tensor)
                u = self.evaluate(x_tensor)[0]
                u = tf.reshape(u, (3, 1))
                metric_norm_squared = tf.matmul(tf.matmul(tf.transpose(u), g), u)
                metric_norm = tf.sqrt(metric_norm_squared)
                return tf.squeeze(metric_norm).numpy()

            # Perform the minimization starting from the best point found in the random search
            result_local = minimize(objective_function, min_point, method='Nelder-Mead')
            best_result = result_local
            '''
            # Perform a global optimization using Differential Evolution
            bounds = [(0, 1), (0, 1), (0, 1)]
            result_global = differential_evolution(objective_function, bounds)
            # Choose the best result between global and local optimization
            if result_global.fun < result_local.fun:
                best_result = result_global
                print(f"Global optimization gave a better result.")
            else:
                best_result = result_local
                print(f"Local optimization gave a better result.")
            '''
            # Get the point with the smallest norm found
            min_norm_final = best_result.fun
            min_point_final = best_result.x

            # Print the results after minimization
            if min_norm_final < 1e-3:
                print(f"Found a zero vector at {min_point_final} with norm {min_norm_final}! Yipeee")
            else:
                print("No zero vector found. :( Boohoo")

            print("Smallest norm found after minimization: " + str(min_norm_final))

if __name__ == '__main__':
    # Generate collocation points within a unit cube
    num_samples = 1000
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)
    # Initialise PINN model
    pinn = PINN()

    # Train the model
    pinn.train(x_collocation, epochs=100, learning_rate=0.001)

    # check for periodicity (should always be true)
    inputs = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
    outputs = pinn.evaluate(inputs)
    print(f"Outputs for [1, 1, 1] and [2, 2, 2]:\n{outputs.numpy()} (these should be exactly the same if the NN has learnt a periodic solution)")

    # check for constant output (should be true for the identity metric)
    random_inputs = np.random.uniform(low=0, high=1, size=(5, 3))
    random_inputs_tensor = tf.convert_to_tensor(random_inputs, dtype=tf.float64)
    random_outputs = pinn.evaluate(random_inputs_tensor).numpy()
    print("Random inputs:")
    print(random_inputs)
    print("Random outputs:")
    print(random_outputs)
    print("(these 5 output vectors should be pretty much the same if the identity metric is used)")

    # plot results
    pinn.plot_learned_1_form()
    
    #Find zero vectors
    pinn.find_zero_vector()
