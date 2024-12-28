import abc
import dataclasses
import numpy as np


@dataclasses.dataclass
class AbstractConvolution(abc.ABC):
    """Abstract class for applying a kernel convolution"""
    kernel : np.ndarray

    @abc.abstractmethod
    def apply(self, image_path : str, output_path : str) -> float:
        """Apply a kernel filter to an image, returns the time it took to apply the filter (not including type conversions, saving the image, etc.)"""
        return
    
    def benchmark(self, image_path : str, output_path : str, iterations : int = 10) -> float:
        """Apply a kernel filter to an image multiple times, returns the average time it took to apply the filter (not including type conversions, saving the image, etc.)"""
        total_time = 0
        for i in range(iterations):
            total_time += self.apply(image_path, output_path)
        return total_time / iterations


