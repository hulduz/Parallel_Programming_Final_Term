import cv2
from convolution_appliers.abstract_convolution import AbstractConvolution
from time import time

class OpenCVConvolution(AbstractConvolution):
    """An OpenCV implementation of a kernel convolution, decently fast"""

    def apply(self, image_path : str, output_path : str):
        """Apply a kernel filter to an image, returns the time it took to apply the filter"""
        # Read the image
        input_image = cv2.imread(image_path)
        input_image = cv2.imread(image_path)
        if input_image is None:
            print(f"Erreur : l'image {image_path} n'a pas pu être chargée. Vérifie le chemin.")
            return  # Ou gérer l'erreur comme tu le souhaites
        else:
            output_image = cv2.filter2D(input_image, -1, self.kernel)


        # Measure the time it takes to apply the filter
        start_time = time()

        # Apply the filter to the image
        output_image = cv2.filter2D(input_image, -1, self.kernel)

        end_time = time()

        # Save the output image
        cv2.imwrite(output_path, output_image)

        return end_time - start_time