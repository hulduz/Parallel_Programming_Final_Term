import pycuda.autoinit # This is needed for initializing CUDA driver  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from time import time
import numpy as np
from PIL import Image

from convolution_appliers.abstract_convolution import AbstractConvolution


class CudaConvolution(AbstractConvolution):
    """An CUDA implementation of a kernel convolution, to be tested"""

    # Compile kernel with slight optimizations
    mod = SourceModule(open("utils/convolution_kernel.cu", "r").read(), options=['--use_fast_math', '-O3'])

    # Get functions
    convolution_kernel = mod.get_function("convolve")

    def apply(self, image_path: str, output_path: str):
        """Apply a kernel filter to an image, returns the time it took to apply the filter"""
        # ensure the image has color channels
        image = Image.open(image_path).convert("RGB") # Possible improvement: support grayscale images, this is just a quick fix, really not optimized

        # Convert the image to a NumPy array
        input_array = np.array(image, dtype=np.uint8)
        output_array = np.empty_like(input_array)

        # Set up parameters for the convolution kernel
        height, width, channels = input_array.shape
        kernel_size = self.kernel.shape[0] # we assume the kernel is a square
        kernel_flat = np.array(self.kernel, dtype=np.float32).flatten()

        # Allocate memory on the device and copy data
        d_input_image = cuda.mem_alloc(input_array.nbytes)
        d_output_image = cuda.mem_alloc(input_array.nbytes)
        d_kernel = cuda.mem_alloc(kernel_flat.nbytes)

        block_size = (32, 32, 1)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1],
        )
        # Used for shared memory, no observable performance difference
        #shared_mem_size = (block_size[0] + 2 * (kernel_size // 2)) * (block_size[1] + 2 * (kernel_size // 2)) * channels


        # Measure the time it takes to apply the filter
        start_time = time()

        cuda.memcpy_htod(d_input_image, input_array)
        cuda.memcpy_htod(d_kernel, kernel_flat)

        self.convolution_kernel(
            d_input_image,
            d_output_image,
            d_kernel,
            np.int32(width),
            np.int32(height),
            np.int32(channels),
            np.int32(kernel_size),
            block=block_size,
            grid=grid_size,
            #shared=shared_mem_size,
        )

        cuda.Context.synchronize()  # Wait for the GPU to finish, to ensure the timing is correct

        cuda.memcpy_dtoh(output_array, d_output_image)
        end_time = time()

        # Save the result as an image
        output_image = Image.fromarray(output_array)
        output_image.save(output_path)

        # Clean up memory
        d_input_image.free()
        d_output_image.free()
        d_kernel.free()

        return end_time - start_time
