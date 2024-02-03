# Credit Card Fraud Detection Model Evaluation

This repository contains Python code for evaluating the performance of different machine learning models on a credit card fraud detection dataset. The code explores the impact of various sampling techniques on model accuracy and recall.

## Data Handling

The dataset used for this analysis is loaded using the pandas library:

```python
import pandas as pd

credit_df = pd.read_csv('Creditcard_data.csv')
```

The dataset has a column named 'Class' representing the target variable (fraud or not fraud).

## Sampling Techniques

To address the issue of class imbalance in the dataset, several sampling techniques are employed:

1. **Random Under-Sampling (S1)**
2. **Random Over-Sampling (S2)**
3. **Under-Sampling using Tomek Links (S3)**
4. **Synthetic Minority Oversampling Technique (SMOTE) (S4)**
5. **NearMiss (S5)**

These techniques are implemented using the `imblearn` library, and each is applied to the dataset before training the models.

## Machine Learning Models

The following machine learning models are used for evaluation:

- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors Classifier
- XGBoost Classifier

**The models are trained and evaluated based on accuracy and recall score both.**

## Model Training and Evaluation

The code includes functions for training models and evaluating their performance:

```python
def train_model(model_name, train_X, train_Y):
    # ... Model initialization and training logic ...

def performance(measure, test_Y, predictions):
    # ... Function to calculate accuracy or recall ...

def evaluate(models, measure):
    # ... Evaluation function using different sampling techniques ...

# Evaluating models based on recall
recall_result_df = evaluate(models, 'recall')

# Evaluating models based on accuracy
accuracy_result_df = evaluate(models, 'accuracy')
```

The evaluation results are stored in Pandas DataFrames (`recall_result_df` and `accuracy_result_df`) for further analysis.

## Why two perforamnce measures ?

 Because analyzing both accuracy and recall is crucial in imbalanced datasets like credit card fraud detection. While accuracy measures overall correctness, recall focuses on correctly identifying instances of the fraud. High recall is essential in fraud detection to minimize missing fraudulent cases. Balancing trade-offs between accuracy and recall is important for informed decision-making and model selection.

## Reproducibility

For reproducibility, a seed value of 42 is used throughout the code.

## Results
After careful evaluation of both scores (accuracy and recall) for all the models , across all the different sampling techniques , we conclude that **Random Forest Classifier** and **XGBoost Classifier** when paired with **(S2)Random Over-Sampling** , **(S3)Under-Sampling using Tomek Links** ,  **(S4)Synthetic Minority Oversampling Technique (SMOTE)** prove to be absolute best models for classifying where output feature is highly imbalanced. The evaluation results are printed and saved as CSV files (`recall_result.csv` and `accuracy_result.csv`). 