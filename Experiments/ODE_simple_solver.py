import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

# Define the neural network model
NN = tf.keras.models.Sequential([
    tf.keras.layers.Input((1,)),  # input layer with 1 input: x
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=64, activation='tanh'),
    tf.keras.layers.Dense(units=32, activation='tanh'),
    tf.keras.layers.Dense(units=1)
])

NN.summary()

class PINN:
    def __init__(self, model):
        self.model = model

    def loss(self, x_init, u_init, x_bound, u_bound, x_collocation):
        # Loss on the initial condition f(0) = 0
        u_init_pred = self.model(x_init)
        init_loss = tf.reduce_mean(tf.square(u_init - u_init_pred))

        # Loss on the boundary condition f(2pi) = 0
        u_bound_pred = self.model(x_bound)
        bound_loss = tf.reduce_mean(tf.square(u_bound - u_bound_pred))

        # Loss on the differential equation f''(x) - sin(x) = 0 at collocation points
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_collocation)
            with tf.GradientTape() as tape1:
                tape1.watch(x_collocation)
                u_collocation = self.model(x_collocation)
            u_x = tape1.gradient(u_collocation, x_collocation)
        u_xx = tape2.gradient(u_x, x_collocation)

        rhs_collocation = tf.sin(x_collocation)
        collocation_loss = tf.reduce_mean(tf.square(u_xx - rhs_collocation))

        total_loss = init_loss + bound_loss + collocation_loss
        return total_loss

    def train(self, x_init, u_init, x_bound, u_bound, x_collocation, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_history = []

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss_value = self.loss(x_init, u_init, x_bound, u_bound, x_collocation)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            loss_history.append(loss_value.numpy())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

        return loss_history

# Generate initial condition data (u(0) = 0)
x_init = np.array([[0.0]], dtype=np.float64)
u_init = np.array([[0.0]], dtype=np.float64)

# Generate boundary condition data (u(2pi) = 0)
x_bound = np.array([[2.0 * np.pi]], dtype=np.float64)
u_bound = np.array([[0.0]], dtype=np.float64)

# Generate collocation points
num_samples_collocation = 1000
x_collocation = np.random.uniform(low=0, high=2 * np.pi, size=(num_samples_collocation, 1))

# Convert to tensors
x_init_tf = tf.convert_to_tensor(x_init, dtype=tf.float64)
u_init_tf = tf.convert_to_tensor(u_init, dtype=tf.float64)
x_bound_tf = tf.convert_to_tensor(x_bound, dtype=tf.float64)
u_bound_tf = tf.convert_to_tensor(u_bound, dtype=tf.float64)
x_collocation_tf = tf.convert_to_tensor(x_collocation, dtype=tf.float64)

# Initialize the PINN
pinn = PINN(NN)

# Train the model
epochs = 6000
learning_rate = 0.001
loss_history = pinn.train(x_init_tf, u_init_tf, x_bound_tf, u_bound_tf, x_collocation_tf, epochs, learning_rate)

# Plot the training loss
plt.figure(figsize=(10, 8))
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Iteration', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('Training Loss', fontsize=15)
plt.show()

# Prediction
test_x = np.linspace(0, 2 * np.pi, 100).astype(np.float64).reshape(-1, 1)
true_f = -np.sin(test_x)
pred_f = NN.predict(test_x).ravel()

# Plot the prediction and true function
plt.figure(figsize=(10, 8))
plt.plot(test_x, true_f, '-k', label='True')
plt.plot(test_x, pred_f, '--r', label='Prediction')
plt.legend(fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('f', fontsize=15)
plt.title('ODE Solution', fontsize=15)
plt.show()
