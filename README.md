# Transformer Encoder for fMRI Classification

This project applies a Transformer Encoder model to classify fMRI data based on participant demographics. The model combines brain imaging data with participant age features to predict characteristics like gender, with the multi-head attention mechanism capturing complex patterns across brain regions. Hyperparameters are tuned using Optuna to maximize predictive performance.

The goal is to use a **Transformer Encoder model** to analyze fMRI data (brain imaging data) and predict demographic traits, specifically **gender** and **age** groups. The Transformer model uses **multi-head attention** to understand correlations between different brain regions in the fMRI data, improving its ability to learn meaningful patterns from brain data.

### Project Steps

1. **Data Loading and Exploration**: Load and examine demographic and fMRI feature data.
2. **Data Preprocessing**: Prepare the data by standardizing age and brain features and encoding categorical variables.
3. **Transformer Model Architecture**: Build a multi-head attention model to learn correlations within the fMRI data.
4. **Training and Hyperparameter Optimization**: Use Optuna for finding the best model settings.
5. **Evaluation and Results**: Analyze model performance with metrics like accuracy, AUC-ROC, precision, recall, and F1-score.

---

## 1. Installation

Install the required packages to set up the environment:

```bash
pip install torch==2.4.0 torchvision optuna-integration pytorch-ranger torch_optimizer nilearn pandas seaborn plotly
```

---

## 2. Data Preparation

### a) Demographic Data (Y)

The demographic data (`pheno.csv`) includes participant IDs, age, gender, and other details. We focus on **age** and **gender**:
```plaintext
participant_id | Age | AgeGroup | Gender
----------------------------------------
sub-pixar123   | 27  | Adult    | F
```

### b) fMRI Features (X)

The brain imaging data, `pixar_features.npz`, contains feature vectors for each participant, representing various brain regions. Each row is a participant, and each column is a brain feature:
- Shape: `(number_of_participants, number_of_features)`

---

## 3. Model Architecture

### Core Components

The model includes several key components to process and combine the brain features with age information for classification:

1. **Age MLP**: Processes age data into embedding vectors, which capture meaningful representations of age.
2. **ROI Encoder**: Encodes fMRI brain data to a higher-dimensional space, preparing it for the attention mechanism.
3. **Multi-Head Attention Transformer Encoder**:
   - **Multi-Head Attention** allows the model to focus on various relationships between brain regions by capturing attention patterns over multiple “heads.” Each head learns a different aspect of the brain feature correlations.
   - **Positional Encoding** is added to the features to give them sequence information, allowing the model to understand spatial relationships between brain regions.
4. **Output Layer**: A single layer that combines the processed features to predict the target (e.g., gender).

### Why Multi-Head Attention?

Multi-head attention is useful here because it:
- Looks at different parts of the input (brain features) to capture various perspectives of how different regions relate to each other.
- Combines all these perspectives for a comprehensive understanding of patterns in brain activity.

---

## 4. Training Process

### Hyperparameter Tuning with Optuna

Optuna suggests optimal hyperparameters like **embedding dimension**, **number of attention heads**, **learning rate**, and **dropout rate**. These parameters directly influence how well the Transformer learns patterns.

### Early Stopping and Optimization

To avoid overfitting, **early stopping** halts training if validation performance stops improving. A **Ranger optimizer** with a cosine annealing learning rate scheduler improves convergence.


## 5. Model Evaluation

After training, the model’s performance is assessed using the following metrics:

- **Accuracy**: The percentage of correct predictions.
- **AUC-ROC**: Measures the model's ability to distinguish between classes (higher is better).
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The ability of the model to identify all positive instances.
- **F1-Score**: A balance between precision and recall.

Sample Evaluation Results:
- **Accuracy**: 0.6452
- **AUC-ROC**: 0.6857
- **Precision**: 0.4706
- **Recall**: 0.8000
- **F1-Score**: 0.5926


