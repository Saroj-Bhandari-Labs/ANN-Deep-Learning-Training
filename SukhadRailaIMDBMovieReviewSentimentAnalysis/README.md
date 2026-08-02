# SukhadRailaIMDBMovieReviewSentimentAnalysis

## Student
Name: Sukhad Raila

## Project Title
IMDB Movie Review Sentiment Analysis using Artificial Neural Networks (ANN)

## Objective
To design, train, and evaluate a simple Artificial Neural Network (ANN) model in PyTorch to classify movie reviews from the IMDB dataset as either positive or negative sentiment.

## Dataset
- **Dataset**: Stanford IMDB Movie Review Dataset (`stanfordnlp/imdb`)
- **Total Samples**: 50,000 movie reviews
- **Splits**: 25,000 training reviews and 25,000 test reviews
- **Vocabulary Size**: 10,000 most frequent words

## ANN Architecture
The model is built using PyTorch with the following sequence of layers:
1. **Embedding Layer**: Maps vocabulary indices (10,000 size) to 16-dimensional dense embedding vectors.
2. **Global Average Pooling**: Computes the average embedding vector across all tokens in each review sequence.
3. **Hidden Layer**: Fully connected (`Linear`) layer with 16 units and `ReLU` activation.
4. **Output Layer**: Fully connected (`Linear`) layer with 1 unit and `Sigmoid` activation for binary classification.

## Results
- **Training Epochs**: 10 epochs
- **Test Accuracy**: **85.38%**
- **Test Loss**: **0.5513**
- **Artifacts Saved**:
  - Model Weights: `model/sentiment_model.pt`
  - Training Plot: `results/training_history.png`
  - Performance Summary Report: `report/report.txt`

## Conclusion
The simple Artificial Neural Network effectively learned sentiment representations from the text embeddings, achieving over 85% accuracy on the test set. The project structure follows all college guidelines cleanly separated into `src/`, `model/`, `results/`, `report/`, and `dataset/` directories.

