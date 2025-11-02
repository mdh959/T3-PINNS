# random_metric.py
import numpy as np
import tensorflow as tf

class RandomMetricGenerator:
    """
    Generate random symmetric matrices and produce a positive-definite metric
    by exponentiating the symmetric matrix (matrix exponential).
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

    def generate_symmetric_matrix(self):
        # Sample from normal, make symmetric
        A = tf.random.normal((3, 3), dtype=tf.float64)
        symmetric_matrix = (A + tf.transpose(A)) / 2.0
        return tf.linalg.expm(symmetric_matrix)

    def generate_metric_tensor(self, batch_size=None):
        sym = self.generate_symmetric_matrix()
        metric = tf.linalg.expm(sym)  # SPD

        # Regularize to avoid near-singular
        eps = tf.constant(1e-6, dtype=tf.float64)
        metric = metric + tf.eye(3, dtype=tf.float64) * eps

        if batch_size is not None:
            metric = tf.expand_dims(metric, axis=0)  # (1, 3, 3)
            metric = tf.repeat(metric, repeats=batch_size, axis=0)

        return metric


    # alias used in some variants
    def generate_new_metric(self):
        return self.generate_metric_tensor()
