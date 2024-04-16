import math
from typing import Dict, Tuple

import cv2
import numpy as np

class DocumentType:
    types: Dict[int, Tuple[int, int, int]]

def get_region_by_doc_type(image: np.ndarray, doc_type: int) -> np.ndarray:
    """Метод возвращает определенную область изобржения, по ее типу. Поддерживается 3 типа документов.

    Args:
        image (np.ndarray): одноканальное изображение
        doc_type (int): тип документа

    Returns:
        Tuple[
            np.ndarray: область изобржения
            np.ndarray: начальные координаты окна на исходном изображении
        ]
    """
    h, w = image.shape[-2:]
    if doc_type == 0:
        image = image[100:300, :int(w // 1.75)]
        new_shape = (600, image.shape[1])
        new_image = (np.ones(new_shape) * 255).astype(np.uint8)
        new_image[:200, :] = image
        return new_image, np.array([0, 100])
    elif doc_type == 1:
        image = image[100:300, :w // 2]
        new_shape = (600, image.shape[1])
        new_image = (np.ones(new_shape) * 255).astype(np.uint8)
        new_image[:200, :] = image
        return new_image, np.array([0, 100])
    elif doc_type == 2:
        image = image[100:300, w // 2:]
        new_shape = (600, image.shape[1])
        new_image = (np.ones(new_shape) * 255).astype(np.uint8)
        new_image[:200, :] = image
        return new_image, np.array([w // 2, 100])

    raise ValueError(f"Unknown type of document <{doc_type}> !!!!")


def classification_preproccesing(image, image_size):
    """_summary_

    Args:
        image (_type_): _description_
        image_size (_type_): (height, width)
    """
    h, w = image.shape[:2]
    image = image[:h // 5, :]
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255
    image = image[None, None, ...]

    return image
