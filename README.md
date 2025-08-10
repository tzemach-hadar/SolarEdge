# SolarEdge Chat Classification

A machine learning project for classifying customer support chat transcripts into product categories using a DistilBERT-based model.

## Overview
This project automates the classification of chat conversations to:
- Reduce response and resolution time for support tickets.
- Automatically detect recurring issues for proactive service.

## Tech Stack
- Python, PyTorch Lightning, Hugging Face Transformers
- Pandas, Matplotlib, Seaborn, spaCy
- scikit-learn for evaluation metrics

## Quick Start
```bash
git clone https://github.com/tzemach-hadar/SolarEdge.git
cd SolarEdge
pip install -r requirements.txt

# run clean data
python3 data_preprocessing/data_cleaning.py

# run analytics
python3 data_preprocessing/data_analysis.py

# run labeling
python3 data_preprocessing/data_labeling.py

# run tokenizer
python3 data_preprocessing/tokenizer.py

# model training 
python3 training/train.py

# model validation
python3 validation/model_validation.py
