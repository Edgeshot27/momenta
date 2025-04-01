# AASIST Model for Audio Classification

This repository contains the implementation of the AASIST model for audio classification, with a focus on speech spoofing detection using the ASVspoof 2019 LA dataset.

## Overview

The **AASIST** model is used for identifying whether an audio sample is bonafide or spoofed. The repository includes the code for training and evaluating the model, as well as a Jupyter notebook for exploratory data analysis and training evaluation.

## Features

- AASIST model implementation for audio classification.
- Uses the ASVspoof 2019 LA dataset for training and testing.
- Data preprocessing, padding, and transformation for uniform input size.
- PyTorch-based model with 1D convolutional layers.
- GPU support for faster training with memory management optimizations.

## Installation

### Requirements

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/aasist-audio-classification.git
    cd aasist-audio-classification
    ```

2. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that you have a CUDA-enabled GPU and appropriate drivers for training with PyTorch on GPU.

### Dataset

The dataset used is the **ASVspoof 2019 LA dataset** .

Run the code cell in jjupyter notebook for downloading the dataset.

After downloading the dataset, extract it and place it in the directory path:



### Training the Model

1. Modify the `data_dir` and `target_length` in the script or Jupyter notebook to point to your dataset directory.

2. Train the model:

    ```python
    from train import train_model
    train_model(data_dir='LA/ASVspoof2019_LA_train/flac', target_length=66230)
    ```

3. Monitor the training process. The model will save the best weights during training, and the results will be stored in the `eval_scores_using_best_dev_model.txt` file.
