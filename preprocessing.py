from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

IMG_SIZE = 128
DEFAULT_POS_DIR = Path("images/positive")
DEFAULT_NEG_DIR = Path("images/negative")
DEFAULT_OUTPUT_DIR = Path("preprocessed_images")
DEFAULT_CSV_FILE = Path("dataset.csv")


def _read_grayscale(image_path: Path) -> np.ndarray:
    image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def _binarize_image(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    thresh = cv.adaptiveThreshold(
        blurred,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        101,
        15,
    )

    kernel = np.ones((4, 4), np.uint8)
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thick_objects = cv.erode(closed, kernel, iterations=2)
    resized = cv.resize(thick_objects, output_size, interpolation=cv.INTER_AREA)

    _, final_binary = cv.threshold(resized, 160, 255, cv.THRESH_BINARY)
    return final_binary


def process_image_to_matrix(image_path: Path | str, output_size: tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> tuple[np.ndarray, np.ndarray]:
    """Read an image and return its binary matrix and binary image."""
    image_path = Path(image_path)
    grayscale = _read_grayscale(image_path)
    final_binary = _binarize_image(grayscale, output_size)
    matrix = (final_binary / 255).astype(np.uint8)
    return matrix, final_binary


def _image_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.*"))


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preprocess_images(
    path_pos: Path | str = DEFAULT_POS_DIR,
    path_neg: Path | str = DEFAULT_NEG_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Preprocess positive and negative images into a single output directory."""
    pos_dir = Path(path_pos)
    neg_dir = Path(path_neg)
    output_dir = Path(output_dir)
    _ensure_directory(output_dir)

    pos_paths = _image_paths(pos_dir)
    neg_paths = _image_paths(neg_dir)
    output_paths: list[Path] = []

    for index, image_path in enumerate(pos_paths + neg_paths):
        _, binary_image = process_image_to_matrix(image_path)
        prefix = "pos" if image_path in pos_paths else "neg"
        output_path = output_dir / f"{prefix}_{index}.png"
        print(f"Processing {image_path} -> {output_path}")
        cv.imwrite(str(output_path), binary_image)
        output_paths.append(output_path)

    return output_paths


def gen_csv(
    source_dir: Path | str = DEFAULT_OUTPUT_DIR,
    csv_file: Path | str = DEFAULT_CSV_FILE,
) -> None:
    """Generate a CSV dataset from previously preprocessed images."""
    source_dir = Path(source_dir)
    csv_file = Path(csv_file)
    image_paths = _image_paths(source_dir)
    rows: list[np.ndarray] = []

    for image_path in image_paths:
        image = _read_grayscale(image_path)
        matrix = (image / 255).astype(np.uint8).flatten()
        label = 1 if image_path.name.startswith("pos_") else 0
        rows.append(np.append(matrix, label))

    array = np.vstack(rows) if rows else np.zeros((0, IMG_SIZE * IMG_SIZE + 1), dtype=np.uint8)
    print(array.shape)
    np.savetxt(csv_file, array, delimiter=",", fmt="%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image datasets and export a CSV file.")
    parser.add_argument("--pos-dir", default=DEFAULT_POS_DIR, help="Path to positive images.")
    parser.add_argument("--neg-dir", default=DEFAULT_NEG_DIR, help="Path to negative images.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Path to save preprocessed images.")
    parser.add_argument("--csv-file", default=DEFAULT_CSV_FILE, help="Output CSV filename.")
    args = parser.parse_args()

    preprocess_images(args.pos_dir, args.neg_dir, args.output_dir)
    gen_csv(args.output_dir, args.csv_file)


if __name__ == "__main__":
    main()
