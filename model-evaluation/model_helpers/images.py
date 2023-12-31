from PIL import Image
import os


def load_images(path):
    """
    Loads all images from a directory and returns them as a list.

    :param path: The path to the directory.
    :return: A list of PIL Image objects.
    """
    images = []
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(path, filename)
            try:
                with Image.open(img_path) as img:
                    images.append(img.copy())  # Copy the image to ensure the file is not left open
            except IOError:
                print(f"Error opening image file {img_path}. Skipping.")
    return images
