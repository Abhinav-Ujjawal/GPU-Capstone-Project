# Import necessary libraries
import tensorflow as tf
import numpy as np

# Define the dimensions of the matrices
MATRIX_SIZE = 1024

# Create two random matrices using NumPy
matrix_a = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
matrix_b = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# Convert matrices to TensorFlow tensors
matrix_a_tensor = tf.constant(matrix_a)
matrix_b_tensor = tf.constant(matrix_b)

# Function to perform matrix multiplication using GPU
def gpu_matrix_multiplication():
    with tf.device('/GPU:0'):
        result = tf.matmul(matrix_a_tensor, matrix_b_tensor)
    return result

# Function to perform matrix multiplication using CPU (for comparison)
def cpu_matrix_multiplication():
    with tf.device('/CPU:0'):
        result = tf.matmul(matrix_a_tensor, matrix_b_tensor)
    return result

# Run the operations and measure time
import time

# GPU computation
start_gpu = time.time()
gpu_result = gpu_matrix_multiplication()
end_gpu = time.time()

# CPU computation
start_cpu = time.time()
cpu_result = cpu_matrix_multiplication()
end_cpu = time.time()

# Print results
print("GPU Computation Time: {:.5f} seconds".format(end_gpu - start_gpu))
print("CPU Computation Time: {:.5f} seconds".format(end_cpu - start_cpu))

# Validate that both results are approximately equal
if np.allclose(gpu_result.numpy(), cpu_result.numpy()):
    print("The GPU and CPU results match!")
else:
    print("There is a discrepancy between GPU and CPU results!")

# Note: Ensure that TensorFlow is installed and a compatible GPU is set up to run this script.
