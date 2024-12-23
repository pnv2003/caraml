import numpy as np

def random_init(input_size: int, output_size: int) -> np.ndarray:
    return np.random.uniform(-1, 1, (input_size, output_size))

def zeros_init(input_size: int, output_size: int) -> np.ndarray:
    return np.zeros((input_size, output_size))

def xavier_init(input_size: int, output_size: int) -> np.ndarray:
    return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)

def he_init(input_size: int, output_size: int) -> np.ndarray:
    return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
