# Matrix Multiplication with TensorFlow and GPU

This project demonstrates the use of GPU acceleration for matrix multiplication using TensorFlow. It compares the computation time for matrix multiplication on a GPU versus a CPU, showcasing the performance benefits of GPU computing.

## Features
- **Matrix Multiplication**: Performs multiplication of two randomly generated 1024x1024 matrices.
- **GPU vs. CPU Comparison**: Measures and compares the execution time of the operation on GPU and CPU.
- **Result Validation**: Ensures that the results from GPU and CPU computations are identical.

## Prerequisites
- Python 3.x
- TensorFlow (latest version preferred)
- A compatible GPU with appropriate drivers and CUDA support

## Installation
1. Install Python if not already installed.
2. Install TensorFlow using pip:
   ```bash
   pip install tensorflow
   ```
3. Ensure that your GPU drivers and CUDA toolkit are correctly set up. Follow TensorFlow's [GPU setup guide](https://www.tensorflow.org/install/gpu) for detailed instructions.

## Usage
1. Clone the repository or copy the script.
2. Run the script using Python:
   ```bash
   python gpu_matrix_multiplication.py
   ```

## Output
The script prints the computation times for both GPU and CPU, along with a message verifying if the results from both computations match. Example output:
```
GPU Computation Time: 0.12345 seconds
CPU Computation Time: 1.23456 seconds
The GPU and CPU results match!
```

## Notes
- Ensure you have a GPU set up; otherwise, the script will default to CPU execution for both cases.
- TensorFlow automatically assigns operations to GPU when available. Explicit device placement is used in this script for demonstration purposes.

## License
This project is open-source and available for educational purposes.
