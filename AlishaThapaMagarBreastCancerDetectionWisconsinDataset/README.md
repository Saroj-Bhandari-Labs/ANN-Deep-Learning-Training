# AlishaThapaMagarBreastCancerDetectionWisconsinDataset

## Student

Name: Alisha Thapa Magar

## Project Title:

Breast Cancer Detection using Artificial Neural Network

## Objective:

To build an Artificial Neural Network (ANN) model that classifies breast tumors as malignant or benign based on cell nucleus measurements from the Wisconsin Breast Cancer dataset.



## Dataset:

Wisconsin Breast Cancer dataset — 569 samples, 30 numeric features (mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension). Target: diagnosis (malignant/benign), label-encoded. The `id` and an empty `Unnamed: 32` column were dropped during preprocessing. Data was split 80/20 into training (455 samples) and testing (114 samples), then standardized using `StandardScaler`.



## ANN Architecture:

Input layer: 30 features

\- Hidden layer 1: 16 neurons, ReLU activation

\- Hidden layer 2: 8 neurons, ReLU activation

\- Output layer: 1 neuron, sigmoid activation

\- Total parameters: 641

\- Optimizer: Adam

\- Loss function: binary cross-entropy

\- Epochs: 50, batch size: 32, validation split: 0.2

## Results:

Test accuracy: 98.25%

Test loss: 0.0773



### Classification report:

\- Precision: 0.97 (benign), 1.00 (malignant)

\- Recall: 1.00 (benign), 0.95 (malignant)

\- F1-score: 0.99 (benign), 0.98 (malignant)

\- Overall accuracy: 98%



## Conclusion:

The ANN model achieved 98.25% accuracy on the test set, correctly classifying almost all malignant and benign tumors. Recall for the malignant class (0.95) was slightly lower than for benign (1.00), meaning a small number of malignant cases were misclassified as benign — the more clinically costly type of error. Future improvements could focus on techniques like class weighting or threshold tuning to reduce false negatives further.

