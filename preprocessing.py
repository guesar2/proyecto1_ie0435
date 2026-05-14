import pathlib

import cv2 as cv
import numpy as np
from pathlib import Path
IMG_SIZE = 128


def process_image_to_matrix(image_path, output_size=(128, 128)):
    """
    Converts an image to a binary matrix using adaptive thresholding 
    to handle uneven lighting and reduce noise.
    """
    # 1. Read high-res grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image.")

    # 2. Smooth camera noise
    blurred = cv.GaussianBlur(img, (5, 5), 0)

    # 3. Adaptive Threshold (High-Res)
    # Background = 255, Objects = 0
    thresh = cv.adaptiveThreshold(
        blurred, 255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY, 
        101, 10
    )

    # 4. Morphological "Closing" & "Erosion"
    # Kernel size 3x3 or 5x5 depends on how thin the clips are in your original photo
    kernel = np.ones((3, 3), np.uint8)
    
    # Fill internal holes (Closing)
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    # Thicken the black objects (Erosion)
    # This ensures thin clips don't disappear during resizing
    thick_objects = cv.erode(closed, kernel, iterations=2)

    # 5. Resize to 128x128
    # INTER_AREA is the gold standard for downsampling
    resized = cv.resize(thick_objects, output_size, interpolation=cv.INTER_AREA)

    # 6. Final snap-to-binary
    # This removes any "fuzziness" created by the resizing process
    _, final_binary = cv.threshold(resized, 127, 255, cv.THRESH_BINARY)
    
    # 7. Convert to 0/1 Matrix (Background = 1)
    matrix = (final_binary / 255).astype(np.uint8)
    return matrix, final_binary

def preprocess_images():
    img_paths = list(Path("imagenes/positivas").glob("*.jpg")) + list(
        Path("imagenes/negativas").glob("*.jpg")
    )
    output_dir = Path("preprocessed_images")
    output_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(img_paths):
        img = cv.imread(str(img_path))
        _, preprocessed_img = process_image_to_matrix(img_path)
        prefix = "pos" if "positivas" in str(img_path) else "neg"
        cv.imwrite(str(output_dir / f"{prefix}_{i}.png"), preprocessed_img)

def gen_csv():
    img_paths = list(Path("preprocessed_images").glob("*.png"))

    array = np.zeros((len(img_paths), IMG_SIZE * IMG_SIZE + 1), dtype=np.uint8)

    for i, img_path in enumerate(img_paths):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

        # Robust binarization: background = 1, object = 0
        matrix = (img / 255).astype(np.uint8)
        label = 1 if "pos" in str(img_path) else 0

        row = matrix.flatten()
        row = np.append(row, label)

        array[i] = row

    print(array.shape)
    np.savetxt("dataset_1.csv", array, delimiter=",", fmt="%d")

if __name__ == "__main__":
    if pathlib.Path("dataset_1.csv").exists():
        print("dataset_1.csv already exists. Skipping preprocessing.")
    else:
        preprocess_images()
        gen_csv()