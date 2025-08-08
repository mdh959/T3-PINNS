import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

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
        g11 =  0.43504112
        g12 = 0.03546557
        g13 = 0.53901665
        g21 = 0.03546557
        g22 = 2.27251859
        g23 = -1.34865049
        g31 = 0.53901665
        g32 = -1.34865049
        g33 = 2.18648953
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

    def plot_learned_1_form(self): # Plot on T3 to visualise results
        # set up the 3D cubes
        fig = plt.figure(figsize=(18, 6))
            
        # create a grid of points in the cube [0,1]x[0,1]x[0,1]
        num_points = 6
        x_vals = np.linspace(0, 1, num_points)
        y_vals = np.linspace(0, 1, num_points)
        z_vals = np.linspace(0, 1, num_points)
            
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # convert points to tensor
        points_tensor = tf.convert_to_tensor(points, dtype=tf.float64)

        # Get the outputs and their derivatives with respect to inputs
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(points_tensor)
            outputs = self.evaluate(points_tensor)
            
            f1_grads = tape.gradient(outputs[:, 0], points_tensor)
            f2_grads = tape.gradient(outputs[:, 1], points_tensor)
            f3_grads = tape.gradient(outputs[:, 2], points_tensor)

        f1_grads_adjusted = tf.concat([f1_grads[:, 0:1] + 1, f1_grads[:, 1:2], f1_grads[:, 2:]], axis=1)
        f2_grads_adjusted = tf.concat([f2_grads[:, 0:1], f2_grads[:, 1:2] + 1, f2_grads[:, 2:]], axis=1)
        f3_grads_adjusted = tf.concat([f3_grads[:, 0:1], f3_grads[:, 1:2], f3_grads[:, 2:3] + 1], axis=1)
            
        # define a function to plot a 3D vector field
        def plot_vector_field(ax, grads, title):
            U = grads[:, 0].numpy().reshape((num_points, num_points, num_points))
            V = grads[:, 1].numpy().reshape((num_points, num_points, num_points))
            W = grads[:, 2].numpy().reshape((num_points, num_points, num_points))
                
            ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])
            ax.set_title(title)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            
        # Plot the vector fields for f1, f2, and f3 derivatives
        ax1 = fig.add_axes([0.05, 0.15, 0.25, 0.7], projection='3d')  # [left, bottom, width, height]
        plot_vector_field(ax1, f1_grads_adjusted, "one-form (a)")

        ax2 = fig.add_axes([0.35, 0.15, 0.25, 0.7], projection='3d')
        plot_vector_field(ax2, f2_grads_adjusted, "one-form (b)")

        ax3 = fig.add_axes([0.65, 0.15, 0.25, 0.7], projection='3d')
        plot_vector_field(ax3, f3_grads_adjusted, "one-form (c)")
            
        plt.tight_layout()
        plt.show()

    def find_zero_vector(self):
        # Define the objective function for minimization (for all 1-forms)
        def objective_function(x):
            x_tensor = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float64)
            g = self.metric_tensor(x_tensor)
            u = self.evaluate(x_tensor)[0] 
            u = tf.reshape(u, (3, 1))
            metric_norm_squared = tf.matmul(tf.matmul(tf.transpose(u), g), u)
            metric_norm = tf.sqrt(metric_norm_squared)
            return tf.squeeze(metric_norm).numpy()

        # Perform a global optimization using Differential Evolution
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        result = differential_evolution(objective_function, bounds)
        # Get the point with the smallest norm found
        min_norm_final = result.fun
        min_point_final = result.x
        print("Smallest norm found after minimization: " + str(min_norm_final), f"at point: {min_point_final}" , f"verification: {objective_function(min_point_final)}")

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
