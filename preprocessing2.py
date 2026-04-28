import cv2 as cv
import numpy as np
from pathlib import Path

IMG_SIZE = 128

def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Estimate background (removes shadows)
    background = cv.GaussianBlur(gray, (31, 31), 0)

    # Normalize illumination
    normalized = cv.divide(gray, background, scale=255)

    # Now threshold works much better
    _, binary = cv.threshold(normalized, 0, 255,
                             cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Resize
    resized = cv.resize(binary, (128, 128), interpolation=cv.INTER_AREA)

    # Convert to 0/1
    # binary_01 = (resized > 0).astype(np.uint8)

    return resized

def preprocess_images():
    img_paths = list(Path("imagenes/positivas").glob("*.jpg")) + list(
        Path("imagenes/negativas").glob("*.jpg")
    )
    output_dir = Path("preprocessed_images")
    output_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(img_paths):
        img = cv.imread(str(img_path))
        preprocessed_img = preprocess_image(img)
        prefix = "pos" if "positivas" in str(img_path) else "neg"
        cv.imwrite(str(output_dir / f"{prefix}_{i}.png"), preprocessed_img)

def gen_csv():
    img_paths = list(Path("preprocessed_images").glob("*.png"))

    array = np.zeros((len(img_paths), IMG_SIZE * IMG_SIZE + 1), dtype=np.uint8)

    for i, img_path in enumerate(img_paths):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        with open("test_bw.txt", "w") as f:
            for row in img:
                f.write(" ".join(map(str, row)) + "\n")
        #img = (img < 128).astype(np.uint8)

        label = 1 if "pos" in str(img_path) else 0
        row = img.flatten()
        row = np.append(row, label)

        array[i] = row

    np.savetxt("preprocessed_data.csv", array, delimiter=",", fmt="%d")

if __name__ == "__main__":
    preprocess_images()
    gen_csv()