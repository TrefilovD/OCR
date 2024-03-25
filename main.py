import argparse
import glob
import os

import cv2

from DocEasyOCR import DocEasyOCRInference


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str, help="директория или путь до изображения")
    parser.add_argument("--save_dir", required=True, type=str, help="путь до директории, в которую буду сохраняться результаты")
    parser.add_argument("--onnx_classificator_path", required=True, type=str, help="путь до onnx модели классификации")
    parser.add_argument("--onnx_detector_path", required=True, type=str, help="путь до onnx модели детекции")
    parser.add_argument("--onnx_recognizer_path", required=True, type=str, help="путь до onnx модели распознавания")
    parser.add_argument("--show", action="store_true", help="сохранение изображений с боксами")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()

    if not os.path.isdir(args.image_path):
        paths = [args.image_path]
    else:
        paths = glob.glob(os.path.join(args.image_path, "*"))

    inference = DocEasyOCRInference(
        onnx_classificator_path=args.onnx_classificator_path,
        onnx_detector_path=args.onnx_detector_path,
        onnx_recognizer_path=args.onnx_recognizer_path
    )

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        name = ".".join(path.split('/')[-1].split(".")[:-1])
        os.makedirs(args.save_dir, exist_ok=True)
        inference.read_text(img, os.path.join(args.save_dir, name), args.show)