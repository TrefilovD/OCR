from typing import Literal, List


class CFG:
    image_height: int = 64
    image_width: int = 128
    resize_square: int = 640

    device: Literal["cpu", "cuda"] = "cpu"
    workers = 0
    batch_size = 8

    #Detector + Recognizer
    decoder: str = 'greedy' # TODO Literal
    # recog_network: str = 'generation2'
    lang: List[str] = ['en','ru']
    character: str = '0123456789!"#$%№&\'()*,-./:;<=>?[\\]+ €₽АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    ignore_char: str = ''
    # ignore_char: str = '0123456789'
    allowlist: str = '0123456789+:АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # important parameters
    beamWidth = 5
    text_threshold: float = 0.2
    link_threshold: float = 0.3
    low_text: float = 0.4
    contrast_ths = 0.1
    adjust_contrast = 0.5
    filter_ths = 0.003
    threshold: float = 0.25
    width_ths: float = 0.1
    optimal_num_chars: int = 10

    # parameters
    min_size: int = 20,
    # canvas_size: float = 800,
    mag_ratio: float = 1
    slope_ths: float = 0.1
    ycenter_ths: float = 0.5
    height_ths: float = 0.5
    y_ths: float = 0.5
    x_ths: float = 1
    add_margin: float = 0.1
    bbox_min_score: float = 0.2
    bbox_min_size: int = 3
    max_candidates: int = 2
    output_format: str = 'standard'

    # utils
    similarity_thresh: float = 0.5
    mark_words: List[str] = [
        "договор", "договор №", "кредитный договор", "кредитный договор №"
        "кредитный",
    ]

    '''
    character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    symbol = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €₽'
    model_path = "./cyrillic_g2.pth"
    separator_list = {}
    cyrillic_lang_list = ['ru']
    package_dir = os.path.dirname(recognition.__file__)

    dict_list = {}
    for lang in cyrillic_lang_list:
        dict_list[lang] = os.path.join(package_dir, 'dict', lang + ".txt")
    '''
