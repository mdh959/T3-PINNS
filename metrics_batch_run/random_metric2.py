import numpy as np
import tensorflow as tf
import random

class RandomMetricGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

    def generate_symmetric_matrix(self):
        # Generate a random 3x3 matrix A
        A = tf.random.normal((3, 3), dtype=tf.float64)
        # Make it symmetric: M = (A + A^T) / 2
        symmetric_matrix = (A + tf.transpose(A)) / 2
        return tf.linalg.expm(symmetric_matrix)

    def partial_derivative(self, tape, u, x):
        """ Compute the partial derivative of u with respect to x """
        du_dx = tape.gradient(u, x)
        if du_dx is None:
            raise ValueError(f"Gradient computation returned None for u={u} and x={x}.")
        return du_dx

    def compute_christoffel_symbols(self, g, g_inv, tape):
        """ Compute Christoffel symbols using the metric tensor and its inverse """
        christoffel = np.zeros((3, 3, 3))  # 3D array to store Γ^i_kl

        for i in range(3):
            for k in range(3):
                for l in range(3):
                    sum_term = 0
                    tape.watch(g)  # Watch the metric tensor
                    for m in range(3):
                        # Compute partial derivatives of the metric tensor
                        g_mk_l = self.partial_derivative(tape, g[m, k], g[l])  # ∂_l g_mk
                        g_ml_k = self.partial_derivative(tape, g[m, l], g[k])  # ∂_k g_ml
                        g_kl_m = self.partial_derivative(tape, g[k, l], g[m])  # ∂_m g_kl

                        sum_term += g_inv[i, m] * (g_mk_l + g_ml_k - g_kl_m)

                    christoffel[i, k, l] = 0.5 * sum_term

        return christoffel

    def compute_scalar_curvature(self, g, tape):
        """ Compute scalar curvature using the Christoffel symbols """
        g_inv = tf.linalg.inv(g)  # Inverse metric tensor
        christoffel = self.compute_christoffel_symbols(g, g_inv, tape)

        scalar_curvature = 0
        for mu in range(3):
            for nu in range(3):
                for lam in range(3):
                    tape.watch(christoffel)
                    term1 = self.partial_derivative(tape, christoffel[lam, mu, nu], lam)  # Γ^λ_{μν,λ}
                    term2 = self.partial_derivative(tape, christoffel[lam, mu, lam], nu)  # Γ^λ_{μλ,ν}

                    term3 = 0
                    term4 = 0
                    for sigma in range(3):
                        term3 += christoffel[sigma, mu, nu] * christoffel[lam, lam, sigma]  # Γ^σ_{μν} Γ^λ_{λσ}
                        term4 += christoffel[sigma, mu, lam] * christoffel[lam, nu, sigma]  # Γ^σ_{μλ} Γ^λ_{νσ}

                    scalar_curvature += g_inv[mu, nu] * (term1 - term2 + term3 - term4)

        return scalar_curvature

    def run(self):
        for i in range(5):
            with tf.GradientTape(persistent=True) as tape:
                metric = self.generate_symmetric_matrix()
                scalar_curvature = self.compute_scalar_curvature(metric, tape)
            print(f"Metric {i+1}:\n{metric.numpy()}")
            print(f"Scalar Curvature {i+1}: {scalar_curvature.numpy()}\n")

# Example usage
generator = RandomMetricGenerator(seed=42)
generator.run()
