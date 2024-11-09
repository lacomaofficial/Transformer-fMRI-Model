# Transformer Encoder for fMRI Classification

This project leverages a **Transformer Encoder** model to classify fMRI data based on participant demographics. The model combines brain imaging data with participant age features to predict characteristics like **gender**, utilizing a **multi-head attention** mechanism to capture complex patterns across brain regions. Hyperparameters are tuned using **Optuna** to enhance predictive performance.

The goal of the project is to use a Transformer Encoder model to analyze fMRI data and predict demographic traits, specifically **gender** and **age** groups. By using multi-head attention, the model is able to understand correlations between different brain regions in the fMRI data, improving its ability to learn meaningful patterns.

 
---

## Project Overview

### Steps

1. **Data Loading and Exploration**: Load and examine demographic and fMRI feature data.
2. **Data Preprocessing**: Prepare the data by standardizing age and brain features, and encoding categorical variables.
3. **Transformer Model Architecture**: Build a multi-head attention model to learn correlations within the fMRI data.
4. **Training and Hyperparameter Optimization**: Use Optuna for finding the best model settings.
5. **Evaluation and Results**: Analyze model performance with metrics like accuracy, AUC-ROC, precision, recall, and F1-score.

---

## 1. Installation

Install the required packages:

```bash
pip install torch==2.4.0 torchvision optuna-integration pytorch-ranger torch_optimizer nilearn pandas seaborn plotly
```

---

## 2. Data Preparation

### a) Demographic Data (Y)

The demographic data (`pheno.csv`) includes participant IDs, age, gender, and other details. We focus on **age** and **gender**:

| participant_id | Age | AgeGroup | Gender |
|----------------|-----|----------|--------|
| sub-pixar123   | 27  | Adult    | F      |

### b) fMRI Features (X)

The brain imaging data, `pixar_features.npz`, contains feature vectors for each participant, representing various brain regions. Each row represents a participant, and each column represents a brain feature:
- Shape: `(number_of_participants, number_of_features)`

---

## 3. Model Architecture

### Core Components

The model combines brain features with age information for classification. Key components:

1. **Age MLP**: Processes age data into embedding vectors to capture meaningful age representations.
2. **ROI Encoder**: Encodes fMRI brain data into a higher-dimensional space, preparing it for the attention mechanism.
3. **Multi-Head Attention Transformer Encoder**:
   - **Multi-Head Attention** captures relationships between brain regions by focusing on patterns across multiple “heads.” Each head learns a different aspect of the brain feature correlations.
   - **Positional Encoding** adds sequence information to features, helping the model understand spatial relationships between brain regions.
4. **Output Layer**: A single layer that combines the processed features to predict the target (e.g., gender).

---

### Why Multi-Head Attention?

Multi-head attention helps by:
- Examining various aspects of the input (brain features) to capture different relationships between brain regions.
- Combining these perspectives for a more comprehensive understanding of patterns in brain activity.

---

## 4. Training Process

### Hyperparameter Tuning with Optuna

Optuna suggests optimal hyperparameters such as **embedding dimension**, **number of attention heads**, **learning rate**, and **dropout rate**. These parameters directly affect the Transformer’s learning capacity.

### Early Stopping and Optimization

To avoid overfitting, **early stopping** halts training if validation performance doesn’t improve. A **Ranger optimizer** with a cosine annealing learning rate scheduler improves convergence.

---

## 5. Model Evaluation

After training, the model’s performance is assessed with the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **AUC-ROC**: Measures the model's ability to distinguish between classes.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The model’s ability to identify all positive instances.
- **F1-Score**: A balance between precision and recall.

Sample Evaluation Results:

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.6452 |
| AUC-ROC    | 0.6857 |
| Precision  | 0.4706 |
| Recall     | 0.8000 |
| F1-Score   | 0.5926 |

