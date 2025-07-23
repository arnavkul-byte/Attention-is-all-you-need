# Transformer Model for Machine Translation

This repository contains a from-scratch implementation of the Transformer model, as introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The model is built using PyTorch and is designed for neural machine translation (NMT).

The code is structured to be clear, modular, and easy to follow, making it a valuable resource for understanding the inner workings of the Transformer architecture.

## ✨ Features

* **Complete Encoder-Decoder Architecture**: Full implementation of the Transformer's encoder and decoder stacks.
* **Multi-Head Attention**: Scaled Dot-Product Attention mechanism with multiple heads.
* **Positional Encoding**: Sinusoidal positional encodings to inject sequence order information.
* **Custom Dataset Handling**: Efficient data loading and preprocessing using a custom PyTorch `Dataset` (`BilingualDataset`).
* **Dynamic Tokenizer Building**: Automatically builds `WordLevel` tokenizers from your dataset using the Hugging Face `tokenizers` library.
* **Training & Validation**: A complete training script with a validation loop, checkpointing, and progress bars via `tqdm`.
* **Inference Script**: A ready-to-use script to translate new sentences using a trained model.
* **Configuration Management**: Centralized `config.py` for easy management of hyperparameters and paths.
* **TensorBoard Integration**: Logs training loss and validation metrics (CER, WER, BLEU) for experiment tracking.

## 🏛️ Architecture

This implementation follows the original Transformer architecture. The model consists of an **Encoder** stack and a **Decoder** stack.

* The **Encoder** maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$.
* The **Decoder**, given $\mathbf{z}$, generates an output sequence $(y_1, ..., y_m)$ one element at a time, using the previously generated symbols as additional input.

<p align="center">
  <img src="https://raw.githubusercontent.com/google-research/tensor2tensor/master/tensor2tensor/visualization/transformer.png" alt="Transformer Architecture Diagram" width="400"/>
  <br>
  <em>The Transformer model architecture from the original paper.</em>
</p>

## 📂 Project Structure

The repository is organized into several key files:

```

.
├── Model.py              \# Contains all PyTorch nn.Module classes for the Transformer architecture
├── Dataset.py            \# Defines the custom BilingualDataset for data loading and preprocessing
├── train.py              \# Main script to handle dataset loading, tokenizer building, and model training
├── inference.py          \# Script for running translation on new sentences with a trained model
├── config.py             \# Central configuration file for hyperparameters and paths
└── README.md             \# This file

````

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

Make sure you have Python 3.8+ installed.

### 2. Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
````

You will need a `requirements.txt` file. Here are the necessary packages based on your code:

**`requirements.txt`**:

```
torch
torchvision
torchaudio
datasets
tokenizers
torchmetrics
tqdm
tensorboard
```

### 3\. Configuration

All hyperparameters, file paths, and dataset settings can be modified in the `config.py` file. The default settings are configured to train an English (`en`) to Italian (`it`) translation model using the `opus_books` dataset.

```python
# config.py
def get_config():
  return {
      'batch_size': 8,
      'num_epochs': 20,
      'lr': 10**-4,
      'datasource': 'opus_books',
      'lang_src': 'en',
      'lang_tgt': 'it',
      'seq_len': 350,       # Reduced for faster training on standard hardware
      'd_model': 512,
      # ... other settings
  }
```

## ⚙️ Usage

### 1\. Training the Model

To start the training process, simply run the `train.py` script.

```bash
python train.py
```

The script will:

1.  Download the dataset from Hugging Face Hub.
2.  Build and save tokenizers for the source and target languages if they don't already exist.
3.  Initialize the Transformer model.
4.  Start the training loop, saving model checkpoints after each epoch.
5.  Log training loss and validation metrics to a TensorBoard instance.

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir runs
```

### 2\. Performing Inference (Translation)

Once a model is trained, you can use `inference.py` to translate sentences.

```bash
# Translate a default sentence ("Hello World.")
python inference.py

# Translate a custom sentence
python inference.py "This is a test sentence."
```

The script also allows you to translate a sentence from the validation set by providing its index:

```bash
# Translate the 10th sentence in the validation set and compare with the ground truth
python inference.py 10
```

The output will look like this:

```
Using device: cuda
      ID: 10
  SOURCE: I have never seen a man look so helpless.
  TARGET: Non ho mai visto un uomo così impotente.
PREDICTED: Non ho mai visto un uomo così indifeso . [EOS]
```

## 🔬 Codebase Breakdown

  * **`Model.py`**: Defines the building blocks of the Transformer: `MultiHeadAttentionBlock`, `FeedForwardBlock`, `EncoderBlock`, `DecoderBlock`, `PositionalEncoding`, `LayerNormalization`, etc. The `build_transformer()` function assembles these blocks into a full model.
  * **`Dataset.py`**: The `BilingualDataset` class is responsible for taking a raw text pair, tokenizing it, adding special tokens (`[SOS]`, `[EOS]`, `[PAD]`), and creating the `encoder_mask` and `decoder_mask` needed for training.
  * **`train.py`**: Orchestrates the entire training pipeline, from data acquisition and tokenization to model training and validation. It also handles model checkpointing.
  * **`config.py`**: A clean way to manage all model and training parameters without hardcoding them into the scripts.
  * **`inference.py`**: Provides a straightforward example of how to load a trained model and use it for greedy-decoding to generate translations.

## 📜 Acknowledgments

This project is a personal implementation based on the concepts presented in the following paper:

  * Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. **Attention Is All You Need.** *In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017).*

## 📄 License

This project is open-source. Please feel free to use, modify, and distribute the code. A standard MIT License is recommended if you wish to add one.

```
```
