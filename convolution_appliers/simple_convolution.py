import numpy as np
from PIL import Image
from convolution_appliers.abstract_convolution import AbstractConvolution
from time import time

class SimpleConvolution(AbstractConvolution):
    """A python implementation of a kernel convolution, extremely slow"""

    def convolve(self, image : np.ndarray):
        """Apply a convolution on an image array"""
        
        image_height, image_width, image_channels = image.shape
        
        # we assume the kernel is a square
        kernel_size = self.kernel.shape[0]
        kernel_padding = kernel_size // 2

        # create an output image array, without the borders where the kernel cannot be applied
        output_image = np.zeros((image_height-kernel_size+1, image_width-kernel_size+1, image_channels))

        for i in range(kernel_padding, image_height - kernel_padding - 1):
            for j in range(kernel_padding, image_width - kernel_padding - 1):
                
                # get the current image patch (kernel_size x kernel_size)
                image_patch = image[i-kernel_padding:i+kernel_padding+1, j-kernel_padding:j+kernel_padding+1]

                # apply the kernel to the image patch
                for channel in range(image_channels):
                    output_image[i - kernel_padding, j - kernel_padding, channel] = (self.kernel * image_patch[:, :, channel]).sum()

        # clip the values to be between 0 and 255
        output_image = np.clip(output_image, 0, 255)

        return output_image
    
    def apply(self, image_path : str, output_path : str):
        """Apply a kernel filter to an image, returns the time it took to apply the filter"""

        # Ensure the image has color channels
        input_image = Image.open(image_path).convert("RGB")

        # Convert the image to a NumPy array
        image_array = np.array(input_image)

        # Measure the time it takes to do the convolution
        start_time = time()

        # Apply the convolution to the image
        convolved_image_array = self.convolve(image_array)

        end_time = time()

        # Convert the result back to an image
        output_image = Image.fromarray(convolved_image_array.astype(np.uint8))

        output_image.save(output_path)

        return end_time - start_time

