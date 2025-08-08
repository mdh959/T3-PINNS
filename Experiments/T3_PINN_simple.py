import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the floating point precision
tf.keras.backend.set_floatx('float64')

class SineActivation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2 * np.pi * inputs), tf.cos(2 * np.pi * inputs)], 1)

class PINN:
    def __init__(self):
        # Define the neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,)),  # Input layer with 3 inputs: x1, x2, x3
            SineActivation(),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=64, activation='tanh'),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=3)  # Output layer for f1, f2, f3
        ])

    def partial_derivative(self, tape, u, x, dim):
        du_dx = tape.gradient(u, x)
        return du_dx[:, dim]  # extracting the partial derivative w.r.t specified dimension

    def hodge_star_1_form(self, u): # Hodge star on a 1-form
        u1, u2, u3 = u[:, 0:1], u[:, 1:2], u[:, 2:3]

        # Hodge star operation
        star_u = tf.concat([
            u1,    # coefficient of dx2 ^ dx3
            -u2,   # coefficient of dx1 ^ dx3
            u3     # coefficient of dx1 ^ dx2
        ], axis=1)

        return star_u # Output is a 2-form

    def hodge_star_2_form(self, d_u): # Hodge star on a 2-form
        df2_dx3 = d_u[:, 3:4]
        df3_dx2 = d_u[:, 5:6]
        df3_dx1 = d_u[:, 4:5]
        df1_dx3 = d_u[:, 1:2]
        df1_dx2 = d_u[:, 0:1]
        df2_dx1 = d_u[:, 2:3]

        # compute f1_prime, f2_prime, f3_prime
        f1_prime = df2_dx3 - df3_dx2
        f2_prime = df3_dx1 - df1_dx3
        f3_prime = df1_dx2 - df2_dx1

        # construct the 1-form (output)
        star_d_u = tf.concat([
            f1_prime,   # coefficient of dx1
            f2_prime,   # coefficient of dx2
            f3_prime    # coefficient of dx3
        ], axis=1)

        return star_d_u # Output is a 1-form

    def exterior_derivative_1_form(self, tape, u, x): # Exterior derivative on a 1-form
        du1_dx2 = self.partial_derivative(tape, u[:, 0], x, 1)  # Partial derivative with respect to x2
        du1_dx3 = self.partial_derivative(tape, u[:, 0], x, 2)  # Partial derivative with respect to x3
        du2_dx1 = self.partial_derivative(tape, u[:, 1], x, 0)  # Partial derivative with respect to x1
        du2_dx3 = self.partial_derivative(tape, u[:, 1], x, 2)  # Partial derivative with respect to x3
        du3_dx1 = self.partial_derivative(tape, u[:, 2], x, 0)  # Partial derivative with respect to x1
        du3_dx2 = self.partial_derivative(tape, u[:, 2], x, 1)  # Partial derivative with respect to x2

        d_u = tf.stack([du1_dx2, du1_dx3, -du2_dx1, du2_dx3, du3_dx1, -du3_dx2], axis=1)

        return d_u # Output is a 2-form

    def star_derivative_2_form(self, tape, u, x): # Hodge star on a 3-form (output of exterior derivative on 2-form)
        du1_dx1 = self.partial_derivative(tape, u[:, 0], x, 0)
        du2_dx2 = self.partial_derivative(tape, u[:, 1], x, 1)
        du3_dx3 = self.partial_derivative(tape, u[:, 2], x, 2)
        return du1_dx1 + du2_dx2 + du3_dx3 # Output is a 0-form

    def derivative_function(self, tape, u, x): # Exterior derivative on 0-form
        df_dx1 = self.partial_derivative(tape, u, x, 0)  # Partial derivative with respect to x1
        df_dx2 = self.partial_derivative(tape, u, x, 1)  # Partial derivative with respect to x2
        df_dx3 = self.partial_derivative(tape, u, x, 2)  # Partial derivative with respect to x3
        return tf.stack([df_dx1, df_dx2, df_dx3], axis=1) # Output is a 1-form

    def loss(self, x_collocation):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_collocation)
            # Evaluate the model at the collocation points
            u = self.model(x_collocation)

            # Calculate d hodge_star d hodge_star (LH term)
            hodge_star_u = self.hodge_star_1_form(u)
            hodge_star_d_hodge_star_u = self.star_derivative_2_form(tape, hodge_star_u, x_collocation)
            d_hodge_star_d_hodge_star_u = self.derivative_function(tape, hodge_star_d_hodge_star_u, x_collocation)

            # Calculate hodge_star d hodge_star d (RH term)
            d_u = self.exterior_derivative_1_form(tape, u, x_collocation)
            hodge_star_d_u = self.hodge_star_2_form(d_u)
            d_hodge_star_d_u = self.exterior_derivative_1_form(tape, hodge_star_d_u, x_collocation)
            hodge_star_d_hodge_star_d_u = self.hodge_star_2_form(d_hodge_star_d_u)

            # Sum RH and LH terms of PDE
            sum_tensor =  hodge_star_d_hodge_star_d_u + d_hodge_star_d_hodge_star_u 

            # Compute the loss based on sum_tensor
            loss = tf.reduce_mean(tf.square(sum_tensor))
            norm_factor = tf.reduce_sum(tf.abs(u))
            normalised_loss = loss / norm_factor

        return normalised_loss

    def train(self, x_collocation, epochs, learning_rate): # train NN
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                normalised_loss = self.loss(x_collocation)
            grads = tape.gradient(normalised_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) 
            if epoch % 100 == 0:
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

if __name__ == '__main__':
    # Generate collocation points within a unit cube
    num_samples = 1000
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    # Initialise PINN model
    pinn = PINN()

    # Train the model
    pinn.train(x_collocation, epochs=100, learning_rate=0.001)

    # Periodicity verification
    inputs = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
    outputs = pinn.evaluate(inputs)
    print(f"Outputs for [1, 1, 1] and [2, 2, 2]:\n{outputs.numpy()}")
    # Plot the learned 1-form
    pinn.plot_learned_1_form()
