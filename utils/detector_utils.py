import cv2
import numpy as np
from typing import Tuple

from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
from easyocr.utils import reformat_input
from easyocr.detection import get_textbox

from utils.prepare_images import export_detected_regions


def preproccesing(
        img: np.ndarray,
        resize_square: int,
        mag_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Read input image
    img = img[:img.shape[0] // 2, :]
    img, _ = reformat_input(img)

    # Resize and normalize input image
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, resize_square, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = x.transpose(2, 0, 1)[None, ...]

    return img, x, (ratio_w, ratio_h)

def postproccesing(out_onnx_detector, img, ratios, text_threshold, link_threshold, low_text):
    ratio_w, ratio_h = ratios
    # Extract score and link maps
    score_text = out_onnx_detector[0, :, :, 0]
    score_link = out_onnx_detector[0, :, :, 1]

    # Post-processing to obtain bounding boxes and polygons
    boxes, polys, mapper = getDetBoxes(score_text, score_link,  text_threshold, link_threshold, low_text)

    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)

    crops = export_detected_regions(
        image=img,
        regions=boxes,
        rectify=True
    )

    return boxes, polys, mapper, crops