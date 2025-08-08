import numpy as np
import random
from sympy import symbols, diag, Matrix, sin
from gravipy.tensorial import *

class RandomMetricGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_metric(self):
        # Define symbolic variables
        r, theta, phi = symbols('r \\theta \\phi')
        
        # Random coefficients for the metric components
        a = random.uniform(0.5, 10)  # Random value for coefficient
        b = random.uniform(0.5, 10)  # Random value for coefficient in radial component
        c = random.uniform(0.5, 10)  # Random value for coefficient in angular components

        # Define the metric in a diagonal form with random coefficients
        metric_matrix = diag(
            1 / (1 - 2 * b / r),  # Radial component
            a * r ** 2,  # Angular component (theta)
            c * r ** 2 * sin(theta) ** 2  # Angular component (phi)
        )
        
        return metric_matrix

# Function to convert the generated symbolic metric to a SymPy matrix for GraviPy
def create_gravipy_metric(metric_matrix):
    return Matrix(metric_matrix)

# Initialize random metric generator
generator = RandomMetricGenerator(seed=42)

# Define symbolic variables
r, theta, phi = symbols('r \\theta \\phi')
C = Coordinates('\chi', [r, theta, phi])  # Include all necessary dimensions

# Loop to generate and compute metrics
for i in range(10):
    # Generate random metric
    random_metric_sympy = generator.generate_random_metric()

    # Create a GraviPy Metric Tensor using the random metric
    g = MetricTensor('g', C, random_metric_sympy)

    # Print metric
    print(f"Randomly generated metric (symbolic form) for metric {i + 1}:")
    print(g(All, All))
    print("\n")

    # Compute Christoffel symbols
    Ga = Christoffel('Ga', g)
    #print(f"Christoffel symbols for metric {i + 1}:")
    #print(Ga(-All, All, All))
    #[;lkprint("\n")

    # Compute Riemann tensor
    Rm = Riemann('Rm', g)
    #print(f"Riemann tensor for metric {i + 1}:")
    #print(Rm(-All, All, All, All))
    #print("\n")

    # Compute Ricci tensor and scalar
    Ri = Ricci('Ri', g)
    #print(f"Ricci tensor for metric {i + 1}:")
    #print(Ri(All, All))
    #print("\n")

    # Randomly generate values for r, theta, and phi
    r_val = random.uniform(1.0, 10.0)  # Random value for r
    theta_val = random.uniform(0, np.pi)  # Random value for theta
    phi_val = random.uniform(0, 2 * np.pi)  # Random value for phi

    # Substitute the random values into the Ricci scalar
    ricci_scalar_value = Ri.scalar().subs({r: r_val, theta: theta_val, phi: phi_val})

    print(f"Ricci scalar for metric {i + 1}:", ricci_scalar_value)
    print("Random metric shape:", random_metric_sympy.shape)
    print("\n" + "="*50 + "\n")
