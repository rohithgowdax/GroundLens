# Dataset Setup

This project uses an augmented version of the [DocVQA dataset](https://www.docvqa.org/) with additional annotations for question priors and answer locations. The augmented dataset will be made available upon publication.

## Requirements

A Hugging Face token with access to the dataset is required. See the [Environment Setup](environment_setup.md) for instructions on configuring your HF token.

## Dataset Structure

The dataset includes:
- Original DocVQA images and questions
- Question prior annotations
- Answer location bounding boxes
- OCR embeddings from ColSmol-500M

## Accessing the Dataset

The dataset is hosted on Hugging Face and can be accessed using the following dataset links:

- https://huggingface.co/datasets/indrehus/docvqa-single-page-questions-answer-ocr-colSmol500M-q-priors
- https://huggingface.co/datasets/indrehus/docvqa-single-page-questions-answer-ocr