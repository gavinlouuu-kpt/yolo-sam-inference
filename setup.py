from setuptools import setup, find_packages

setup(
    name="yolo_sam_inference",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",  # for YOLO
        "transformers",  # for SAM from HuggingFace
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "scikit-image",  # for image processing and metrics
    ],
    author="AI Cytometry Team",
    description="A pipeline for cell segmentation using YOLO and SAM models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 