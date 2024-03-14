import argparse
import time
import os
import glob

import onnxruntime as rt
import cv2
import numpy as np
import easyocr

from easyocr import recognition
from easyocr.utils import reformat_input, get_image_list

from config.cfg import CFG
from utils.detector_utils import preproccesing, postproccesing
from utils.recognizer_utils import get_text
from utils.prediction_utils import postproccesing as pred_post, save_detecting, save_results


class InferenceONNX:
    "Класс, реализующий инференс для onnx модели"

    def __init__(
        self,
        onnx_detector_path: str,
        onnx_recognizer_path: str,
        converter = None,
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ) -> None:
        """
        Args:
            onnx_detector_path (str): путь до onnx модели детектора
            onnx_recognizer_path (str): путь до onnx модели распознавателя
            converter (_type_, optional): конвертер для постобработки выходов из распознавателя. Defaults to None.
            providers (list, optional): тип целевого вычислителя. Defaults to ['CPUExecutionProvider'].
        """
        self.onnx_detector_path = onnx_detector_path
        self.onnx_recognizer_path = onnx_recognizer_path
        self.providers = providers
        self.converter = converter

        self.detector_session = rt.InferenceSession(onnx_detector_path, providers=self.providers)
        self.input_name = self.detector_session.get_inputs()[0].name

        self.recognizer_session = rt.InferenceSession(onnx_recognizer_path, providers=providers)

        if self.converter is None:
            separator_list = {}
            lang_list = CFG.lang
            package_dir = os.path.dirname(recognition.__file__)

            dict_list = {}
            for lang in lang_list:
                dict_list[lang] = os.path.join(package_dir, 'dict', lang + ".txt")
            reader = easyocr.Reader(
                CFG.lang,
                gpu=CFG.device=="cuda",
                quantize=False
            )
            self.converter = reader.converter
            # self.converter = recognition.CTCLabelConverter(CFG.character, separator_list, dict_list)

    def read_text(self, image: np.ndarray, save_path: str, show: bool = False) -> None:
        """Метод выполняет поиск текста на картинке

        Args:
            image (np.ndarray): фрейм
            save_path (str): путь до сохранения результатов
        """
        start_det_time = time.time()
        img, inp, ratios = preproccesing(image, CFG.resize_square, CFG.mag_ratio)

        # Prepare input tensor for inference
        input_name = self.detector_session.get_inputs()[0].name
        inp = {input_name: inp}

        # Run inference and get output
        out_onnx_detector, _ = self.detector_session.run(None, inp)

        try:
            boxes, polys, mapper, crops = postproccesing(out_onnx_detector, img, ratios, CFG.text_threshold, CFG.link_threshold, CFG.low_text)
        except:
            print(f"Warning!!! Failed detector postproccesing")
            return None

        print("detector time: ", time.time() - start_det_time)

        start_rec_time = time.time()

        result = []

        for crop in crops:
            img, img_cv_grey = reformat_input(crop)

            y_max, x_max = img_cv_grey.shape

            horizontal_list = [[0, x_max, 0, y_max]]

            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height=CFG.image_height) # 64 is default value
                result0 = get_text(
                    self.recognizer_session,
                    CFG.character,
                    CFG.image_height,
                    int(max_width),
                    self.converter,
                    image_list,
                    CFG.allowlist,
                    CFG.ignore_char,
                    CFG.decoder,
                    beamWidth = CFG.beamWidth,
                    batch_size = CFG.batch_size,
                    contrast_ths = CFG.contrast_ths,
                    adjust_contrast = CFG.adjust_contrast,
                    filter_ths = CFG.filter_ths,
                    workers = CFG.workers,
                    device = CFG.device
                )
                result += result0
        print("recognizer time: ", time.time() - start_rec_time)

        result_cp = [((coord.astype(int), text, prob)) for (_, text, prob), coord in zip(result, boxes)]

        postproccesing_time = time.time()
        result_cp = pred_post(result_cp, CFG.threshold)
        result_cp = sorted(result_cp, key=lambda x: [x[0][1][1], x[0][1][0]])[:1]
        postproccesing_time = time.time() - postproccesing_time
        print("postproccesing_time", postproccesing_time)

        if show:
            save_detecting(result_cp, image, save_path)
        save_results(result_cp, save_path)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str, help="директория или путь до изображения")
    parser.add_argument("--save_dir", required=True, type=str, help="путь до директории, в которую буду сохраняться результаты")
    parser.add_argument("--onnx_detector_path", required=True, type=str, help="директория или путь до изображения")
    parser.add_argument("--onnx_recognizer_path", required=True, type=str, help="директория или путь до изображения")
    parser.add_argument("--show", action="store_true", help="сохранение изображений с боксами")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()

    if not os.path.isdir(args.image_path):
        paths = [args.image_path]
    else:
        paths = glob.glob(os.path.join(args.image_path, "*"))

    inference = InferenceONNX(
        onnx_detector_path=args.onnx_detector_path,
        onnx_recognizer_path=args.onnx_recognizer_path
    )

    for path in paths:
        img = cv2.imread(path)
        name = ".".join(path.split('/')[-1].split(".")[:-1])
        inference.read_text(img, os.path.join(args.save_dir, name), args.show)
