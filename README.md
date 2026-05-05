# Monet GAN

This project implements a Generative Adversarial Network (GAN) to translate photos into Monet-style paintings.

## Project Structure

- `src/data_setup.py`: Contains the data loading and preprocessing logic, including a custom `ImageDataset` for loading images from flat directories and applying transformations.
- `train.py`: The main training script (currently initializing the dataset loaders).
- `data/`: Directory for storing the dataset.
  - `monet_jpg/`: Contains Monet paintings.
  - `photo_jpg/`: Contains photos to be translated.

## Setup and Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python train.py
   ```