import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="DocEasyOCR",
    py_modules=["DocEasyOCR"],
    version="1.0",
    description="",
    author="Trefilov Dmitry",
    url='https://github.com/TrefilovD/OCR',
    packages=find_packages(include=['DocEasyOCR']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)