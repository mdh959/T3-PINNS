import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# set the floating point precision
tf.keras.backend.set_floatx('float64')

class SineActivation(Layer):
    def __init__(self):
        super(SineActivation, self).__init__()

    def call(self, inputs):
        return tf.concat([tf.sin(2*np.pi*inputs), tf.cos(2*np.pi*inputs)], 1)

class PINN:
    def __init__(self):
        # define the neural network model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input((3,)),  # input layer with 3 inputs: x1, x2, x3
            SineActivation(),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=64, activation='tanh'),
            tf.keras.layers.Dense(units=32, activation='tanh'),
            tf.keras.layers.Dense(units=3)  # output layer for f1, f2, f3
        ])

    def partial_derivative(self, u, x, dim):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u = tf.convert_to_tensor(u, dtype=tf.float64)
            du_dx = tape.gradient(u, x)
            partial_derivative_dim = du_dx[:, dim]  # extracting the partial derivative w.r.t specified dimension
        return partial_derivative_dim

    def hodge_star_1_form(self, u):
        u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert u to a tf.Tensor
        u1, u2, u3 = u[:, 0:1], u[:, 1:2], u[:, 2:3]

        # Hodge star operation
        star_u = tf.concat([
            u1,   # coefficient of dx2 ^ dx3
            -u2,   # coefficient of dx1 ^ dx3
            u3    # coefficient of dx1 ^ dx2
        ], axis=1)

        return star_u

    def hodge_star_2_form(self, d_u):
        d_u = tf.convert_to_tensor(d_u, dtype=tf.float64)  # Convert d_u to a tf.Tensor
        
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

        return star_d_u

    def exterior_derivative_1_form(self, u, x):
        u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert u to a tf.Tensor
        du1_dx2 = self.partial_derivative(u[:, 0], x, 1)  # Partial derivative with respect to x2
        du1_dx3 = self.partial_derivative(u[:, 0], x, 2)  # Partial derivative with respect to x3
        du2_dx1 = self.partial_derivative(u[:, 1], x, 0)  # Partial derivative with respect to x1
        du2_dx3 = self.partial_derivative(u[:, 1], x, 2)  # Partial derivative with respect to x3
        du3_dx1 = self.partial_derivative(u[:, 2], x, 0)  # Partial derivative with respect to x1
        du3_dx2 = self.partial_derivative(u[:, 2], x, 1)  # Partial derivative with respect to x2

        d_u = tf.stack([du1_dx2, du1_dx3, -du2_dx1, du2_dx3, du3_dx1, -du3_dx2], axis=1)

        return d_u

    def star_derivative_2_form(self, u, x):
        u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert u to a tf.Tensor
        du1_dx1 = self.partial_derivative(u[:, 0], x, 0)
        du2_dx2 = self.partial_derivative(u[:, 1], x, 1)
        du3_dx3 = self.partial_derivative(u[:, 2], x, 2)
        f = (du1_dx1 + du2_dx2 + du3_dx3)
        return f

    def derivative_function(self, u, x):
        u = tf.convert_to_tensor(u, dtype=tf.float64)  # Convert f to a tf.Tensor
        df_dx1 = self.partial_derivative(u, x, 0)  # Partial derivative with respect to x1
        df_dx2 = self.partial_derivative(u, x, 1)  # Partial derivative with respect to x2
        df_dx3 = self.partial_derivative(u, x, 2)  # Partial derivative with respect to x3
        der_du = tf.stack([
            df_dx1,df_dx2,df_dx3
        ], axis=1)
        return der_du

    def loss(self, x_collocation):
        # evaluate the model at the collocation points
        u = self.model(x_collocation)

        # calculate d hodge_star d hodge_star (LH term)
        hodge_star_u = self.hodge_star_1_form(u)
        hodge_star_d_hodge_star_u = self.star_derivative_2_form(hodge_star_u, x_collocation)
        d_hodge_star_d_hodge_star_u = self.derivative_function(hodge_star_d_hodge_star_u, x_collocation)

        # calculate hodge_star d hodge_star d (RH term)
        d_u = self.exterior_derivative_1_form(u, x_collocation)
        hodge_star_d_u = self.hodge_star_2_form(d_u)
        d_hodge_star_d_u = self.exterior_derivative_1_form(hodge_star_d_u, x_collocation)
        hodge_star_d_hodge_star_d_u = self.hodge_star_2_form(d_hodge_star_d_u)
            
        # sum RH and LH terms of pde
        sum_tensor = hodge_star_d_hodge_star_d_u + d_hodge_star_d_hodge_star_u 
            
        # Compute the loss based on sum_tensor
        loss = tf.reduce_mean(tf.square(sum_tensor))
        norm_factor = tf.reduce_sum(tf.abs(u))
        normalised_loss = loss/ norm_factor

        return normalised_loss

    def train(self, x_collocation, epochs, learning_rate):
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

if __name__ == '__main__': 
    # generate collocation points within a unit cube [0, 1] x [0, 1] x [0, 1]
    num_samples = 100
    x_collocation = np.random.uniform(low=0, high=1, size=(num_samples, 3))
    x_collocation = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

    # initialise PINN model
    pinn = PINN()
    
    # train the model
    pinn.train(x_collocation, epochs = 100, learning_rate = 0.001)

    # evaluate the model on specific points to check for periodicity
    inputs = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
    outputs = pinn.evaluate(inputs)
    print(f"Outputs for [1, 1, 1] and [2, 2, 2]:\n{outputs.numpy()}")

    # generate 5 random inputs to test if the functions learnt are constants
    random_inputs = np.random.uniform(low=0, high=1, size=(5, 3))
    random_inputs_tensor = tf.convert_to_tensor(random_inputs, dtype=tf.float64)
    random_outputs = pinn.evaluate(random_inputs_tensor).numpy()
    resulting_arrays = random_outputs / random_inputs
    print("Resulting arrays from element-wise division of outputs by inputs:")
    print(resulting_arrays)
