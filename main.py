import os
import csv
import cv2
import numpy as np

# Convolution modules
from convolution_appliers.opencv_convolution import OpenCVConvolution
from convolution_appliers.cuda_convolution import CudaConvolution
from convolution_appliers.simple_convolution import SimpleConvolution

# Utility functions
from utils.image_resizer import resize_images_incrementally
from utils.path_manager import generate_output_path

def generate_gaussian_kernel(kernel_size: int = 9, sigma: float = 0):
    """Generate a 2D Gaussian filter."""
    one_d_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    return np.outer(one_d_kernel, one_d_kernel.T)

def setup_convolution_methods(kernel):
    """Configure different convolution methods with the specified filter."""
    return {
        "opencv": OpenCVConvolution(kernel),
        "cuda": CudaConvolution(kernel),
        "simple": SimpleConvolution(kernel),
    }

def perform_benchmark(images, methods, output_dir, results_file):
    """Run performance benchmarks on multiple images for different convolution methods."""
    previous_simple_time = 0

    with open(results_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "OpenCV (s)", "CUDA (s)", "Simple (s)"])

        for image_path in images:
            print(f"Processing image: {image_path}")

            # Timing for OpenCV
            opencv_time = methods["opencv"].benchmark(
                image_path, generate_output_path(image_path, "opencv"), iterations=5
            )

            # Timing for CUDA
            cuda_time = methods["cuda"].benchmark(
                image_path, generate_output_path(image_path, "cuda"), iterations=5
            )

            # Timing for the simple method (if fast enough)
            simple_time = 0
            if previous_simple_time < 10.0:
                print("Skipping simple convolution: time limit exceeded.")
                simple_time = methods["simple"].benchmark(
                    image_path, generate_output_path(image_path, "simple"), iterations=5
                )
                previous_simple_time = simple_time

            # print the results in milliseconds
            print(f"OpenCV: {opencv_time * 1000} ms")
            print(f"CUDA: {cuda_time * 1000} ms")
            print(f"Simple: {simple_time * 1000} ms")
    
            # How much faster is the CUDA implementation than the OpenCV implementation?
            print(f"CUDA is {opencv_time // cuda_time} times faster than OpenCV")
    
            # How much faster is the CUDA implementation than the simple implementation?
            if(simple_time // cuda_time == 0):
                print(f"The simple implementation cannot be compared to the CUDA one")
            else:
                print(
                    f"CUDA is {simple_time // cuda_time} times faster than the simple implementation"
                )
            print("\n\n")
            writer.writerow([image_path, opencv_time, cuda_time, simple_time])

def main():
    # Initial configuration
    input_image_path = "images/image1.jpg"
    output_directory = "output"
    results_csv = "benchmark_results.csv"
    min_size, max_size, step_size = 100, 4000, 100

    # Create the convolution kernel
    gaussian_filter = generate_gaussian_kernel(kernel_size=9)
    assert gaussian_filter.shape[0] % 2 == 1, "The kernel size must be odd."

    # Initialize convolution methods
    convolution_methods = setup_convolution_methods(gaussian_filter)

    # Generate resized images
    resized_image_list = resize_images_incrementally(
        input_image_path, output_directory, min_size, max_size, step_size
    )

    # Run the benchmark
    perform_benchmark(resized_image_list, convolution_methods, output_directory, results_csv)

    print(f"All benchmarks completed. Results saved in {results_csv}.")

if __name__ == "__main__":
    main()
