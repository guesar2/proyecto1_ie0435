import cv2 as cv
import numpy as np
from pathlib import Path

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
        cv.imwrite(str(output_dir / f"{prefix}_{i}.jpg"), preprocessed_img)

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 50, 150)
    resized = cv.resize(edges, (128, 128), interpolation=cv.INTER_AREA)
    # Apply Gaussian blur to reduce noise

    # Apply Canny edge detection

    return resized

def gen_csv():
    img_paths = list(Path("preprocessed_images").glob("*.jpg"))
    array = np.zeros((len(img_paths), 128*128 + 1), dtype=np.uint8)  # +1 for the label
    for i, img_path in enumerate(img_paths):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        label = 1 if "pos" in str(img_path) else 0
        _, binary_img = cv.threshold(img, 20, 1, cv.THRESH_BINARY) # Convert to binary (0 and 1)
        row = binary_img.flatten().astype(np.uint8)
        row = np.append(row, [label])  # Append the label to the end of the array
        array[i] = row
    np.savetxt("preprocessed_data.csv", array, delimiter=",", fmt="%d")

if __name__ == "__main__":
    gen_csv()
    # preprocess_images()
    #img_paths = list(Path("preprocessed_images").glob("*.jpg"))
    #path = img_paths[13]
    #print(path)
    #img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    #print(np.min(img), np.max(img))
    #with open("test_bw.txt", "w") as f:
    #    for row in img:
    #        f.write(" ".join(map(str, row)) + "\n")
    #_, binary_img = cv.threshold(img, 20, 1, cv.THRESH_BINARY)
    #with open("test_array.txt", "w") as f:
    #    for row in binary_img:
    #        f.write(" ".join(map(str, row)) + "\n")
    #arr = binary_img.flatten().astype(int)
    #arr = np.append(arr, [0])  # Append a zero to the end
    #print(arr.shape)