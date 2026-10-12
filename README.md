# Transformers for Movie Review Sentiment Analysis

## Project Overview

This project implements a Transformer-based model for sentiment analysis using the IMDB movie review dataset. The model classifies movie reviews into two sentiment categories: **positive** and **negative**.

The project uses PyTorch to build and train a custom Transformer model and the BERT tokenizer to preprocess text data.

## Objectives

- Prepare and preprocess the IMDB movie review dataset.
- Tokenize reviews using the BERT tokenizer.
- Build a custom Transformer model for sentiment classification.
- Train and evaluate the model using PyTorch.
- Measure performance using validation and test accuracy.

## Dataset

The project uses the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), containing positive and negative movie reviews.

The dataset was divided into three sets:

- **Training set:** 22,500 reviews
- **Validation set:** 2,500 reviews
- **Test set:** 25,000 reviews

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- BERT Tokenizer (`bert-base-uncased`)
- NumPy
- Jupyter Notebook

## Model Architecture

The model is a custom Transformer-based neural network implemented using PyTorch.

Its main components include:

- Token Embeddings
- Positional Embeddings
- Multi-Head Self-Attention
- Feed-Forward Networks
- Transformer Blocks
- Mean Pooling
- Classification Head

The model predicts whether a movie review expresses a positive or negative sentiment.

## Training Configuration

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| Number of Transformer Layers | 4 |
| Number of Attention Heads | 4 |
| Maximum Sequence Length | 128 |
| Batch Size | 32 |
| Number of Epochs | 5 |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Loss Function | Cross-Entropy Loss |

## Results

The model achieved the following accuracy:

| Evaluation Metric | Accuracy |
|---|---:|
| Final Validation Accuracy | 79.12% |
| Test Accuracy | 77.60% |

## Project Structure

```text
transformers-movie-review-sentiment-analysis/
│
├── SentimentScope_starter.ipynb
├── README.md
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/Haila_Abdullah/transformers-movie-review-sentiment-analysis.git
   ```

2. Navigate to the project directory:

   ```bash
   cd transformers-movie-review-sentiment-analysis
   ```

3. Install the required libraries:

   ```bash
   pip install torch transformers numpy jupyter
   ```

4. Download the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and extract it according to the folder structure used in the notebook.

5. Open the notebook:

   ```bash
   jupyter notebook SentimentScope_starter.ipynb
   ```

6. Run the notebook cells in order to preprocess the data, train the model, and evaluate its performance.

## Key Takeaways

- Learned how to preprocess text for Transformer models.
- Implemented a custom PyTorch Dataset and DataLoaders.
- Applied BERT tokenization to movie reviews.
- Built a Transformer architecture for binary sentiment classification.
- Trained and evaluated a deep learning model.
- Achieved 77.60% test accuracy on the IMDB dataset.

## Future Improvements

- Experiment with different learning rates and model configurations.
- Increase the number of training epochs.
- Compare the custom Transformer with pretrained models such as BERT.
- Improve classification accuracy through hyperparameter tuning.

