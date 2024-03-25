import argparse
import time
import os
import glob

from typing import Optional

import onnxruntime as rt
import cv2
import numpy as np
import easyocr

from easyocr import recognition
from easyocr.utils import reformat_input, get_image_list

from config.cfg import CFG
from utils.classification_utils import get_region_by_doc_type, classification_preproccesing
from utils.detector_utils import preproccesing, postproccesing
from utils.recognizer_utils import get_text
from utils.prediction_utils import postproccesing as pred_post, save_detecting, save_results


class DocEasyOCRInference:
    "Класс, реализующий инференс для onnx модели"

    def __init__(
        self,
        onnx_classificator_path: str,
        onnx_detector_path: str,
        onnx_recognizer_path: str,
        converter = None,
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ) -> None:
        """
        Args:
            onnx_classification (str):
            onnx_detector_path (str): путь до onnx модели детектора
            onnx_recognizer_path (str): путь до onnx модели распознавателя
            converter (_type_, optional): конвертер для постобработки выходов из распознавателя. Defaults to None.
            providers (list, optional): тип целевого вычислителя. Defaults to ['CPUExecutionProvider'].
        """
        self.onnx_classificator_path = onnx_classificator_path
        self.onnx_detector_path = onnx_detector_path
        self.onnx_recognizer_path = onnx_recognizer_path
        self.providers = providers
        self.converter = converter

        self.classificator_session = rt.InferenceSession(onnx_classificator_path, providers=self.providers)

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

    def read_text(
            self,
            image: np.ndarray,
            save_path: str,
            show: bool = False,
            use_classification: bool = True,
            doc_type: Optional[int] = None
        ) -> None:
        """Метод выполняет поиск текста на картинке

        Args:
            image (np.ndarray): фрейм
            save_path (str): путь до сохранения результатов
            show (bool): сохранение изображения с выделенным номером и результатов в файл .txt
            use_classification (bool): флаг для предварительной классификации документа
            doc_type (Optional[int]): номер документа. Необязательный параметр.
        """
        # Classification inference
        if doc_type:
            image_region, offset = get_region_by_doc_type(image, doc_type)
        elif use_classification:
            classification_time = time.time()
            input_name = self.classificator_session.get_inputs()[0].name
            input_shape = self.classificator_session.get_inputs()[0].shape[-2:]
            inp = {input_name: classification_preproccesing(image, input_shape)}
            out_onnx_classificator = self.classificator_session.run(None, inp)

            image_region, offset = get_region_by_doc_type(image, out_onnx_classificator[0][0].argmax())
            print("classificator time: ", time.time() - classification_time)
        else:
            image_region = image.copy()
            offset = np.array([0, 0])

        start_det_time = time.time()
        img, inp, ratios = preproccesing(image_region, CFG.resize_square, CFG.mag_ratio)

        # Prepare input tensor for inference
        input_name = self.detector_session.get_inputs()[0].name
        inp = {input_name: inp}

        # Run inference and get output
        out_onnx_detector, _ = self.detector_session.run(None, inp)

        try:
            boxes, polys, mapper, crops = postproccesing(out_onnx_detector, img, ratios, CFG.text_threshold, CFG.link_threshold, CFG.low_text)
        except:
            boxes = []
            crops = []

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
        result_cp_p = pred_post(result_cp, CFG.threshold)
        result_cp_p = sorted(result_cp_p, key=lambda x: [x[0][1][1], x[0][1][0]])[:1]
        postproccesing_time = time.time() - postproccesing_time
        print("postproccesing_time", postproccesing_time)

        for i in range(len(result_cp_p)):
            result_cp_p[i][0] += offset

        if show:
            save_detecting(result_cp_p, image, save_path)
        save_results(result_cp_p, save_path)

