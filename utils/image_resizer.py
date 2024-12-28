import cv2

def resize_images_incrementally(image_path : str, output_folder : str, initial_size : int, final_size : int, step : int) -> list:
    """Resize an image to multiple sizes, saving each resized image to the output folder and returning the paths to the resized images"""
    img = cv2.imread(image_path)

    current_size = initial_size
    output_paths = []
    while current_size <= final_size:
        resized_img = cv2.resize(img, (current_size, current_size))
        output_path = f"{output_folder}/resized_{current_size}.jpg"
        cv2.imwrite(output_path, resized_img)

        print(f"Image resized to {current_size}x{current_size} and saved to {output_path}")

        current_size += step
        output_paths.append(output_path)

    return output_paths