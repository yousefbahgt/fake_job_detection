# Fake Job Posting Detection

A machine learning project for detecting fraudulent job postings using
classical machine learning, TF-IDF-based NLP, class-imbalance handling,
ensemble learning, and frozen Transformer feature extraction.

## Project Overview

Online job platforms can contain fraudulent postings designed to mislead
applicants or obtain personal and financial information. This project
formulates fake job detection as a **binary classification** problem:

-   `0` → Legitimate
-   `1` → Fraudulent

The project uses the **EMSCAD** fake job postings dataset containing
**17,880 job postings**, with approximately **4.8% fraudulent
postings**.

Because the dataset is highly imbalanced, model evaluation focuses on
**Precision, Recall, F1-score, and ROC-AUC**, rather than relying on
Accuracy alone.

## Dataset

The dataset contains job-posting information such as:

-   Job title
-   Location
-   Department
-   Salary range
-   Company profile
-   Description
-   Requirements
-   Benefits
-   Employment type
-   Required experience
-   Required education
-   Industry
-   Function
-   Telecommuting
-   Company logo availability
-   Questions availability

The original dataset is not included in this repository.

## Data Cleaning & Feature Engineering

The shared preprocessing pipeline includes:

-   Handling missing values using `"Unknown"`
-   Text cleaning and normalization
-   Removing HTML tags, URLs, emails, punctuation, and extra whitespace
-   Combining the main job-posting text fields
-   Creating text-length features
-   Creating information-availability features such as salary, location,
    company profile, and benefits availability
-   Creating a `missing_fields_count` feature

All experiments use an **80/20 stratified train/test split with
`random_state=42`**.

## Experiments

### 1. Classical Machine Learning

Compared multiple classical classifiers using TF-IDF text features,
categorical features, and numerical/information-availability features.

**Best F1-score:** Logistic Regression --- **0.720**

### 2. Text Context + Numerical Features

Combined the main text fields:

`title + description + requirements + benefits`

and represented the text using TF-IDF.

**Best model:** Logistic Regression

-   F1: **0.822**
-   Recall: **0.803**
-   Precision: **0.842**
-   ROC-AUC: **0.983**

This showed that richer job-posting text improved performance.

### 3. Class-Imbalance Handling

Applied **Random Over-Sampling (ROS)** and **SMOTE** only to the
training data.

**Best F1-score:** Linear SVC + SMOTE

-   F1: **0.826**
-   Recall: **0.769**
-   Precision: **0.893**
-   ROC-AUC: **0.980**

### 4. Ensemble Learning

Two ensemble strategies were evaluated:

-   Hard Voting
-   Stacking

The Stacking ensemble combines:

-   Logistic Regression + ROS
-   Linear SVC + ROS
-   Random Forest + SMOTE
-   XGBoost + SMOTE

A Logistic Regression model is used as the meta-learner.

**Best overall result: Stacking**

  Metric            Score
  ----------- -----------
  Accuracy          0.987
  Precision         0.950
  Recall            0.769
  F1-score      **0.850**
  ROC-AUC           0.983

### 5. Frozen DistilBERT Feature Extraction

A pre-trained DistilBERT model was used as a **frozen feature
extractor** rather than fine-tuning its weights.

Pipeline:

`Job Text → Tokenization → Frozen DistilBERT → Mean Pooling → Numerical Features → Classifier`

The best configuration used XGBoost with threshold tuning.

-   F1: **0.773**
-   Recall: **0.688**
-   Precision: **0.881**
-   ROC-AUC: **0.982**
-   PR-AUC: **0.866**

The frozen Transformer approach did not outperform the Stacking ensemble
on this dataset.

## Overall Results

  --------------------------------------------------------------------------------
  Experiment    Best Model               F1       Recall    Precision      ROC-AUC
  ------------- -------------- ------------ ------------ ------------ ------------
  Classical ML  Logistic              0.720        0.734        0.706        0.946
                Regression                                            

  Text +        Logistic              0.822        0.803        0.842        0.983
  Numerical     Regression                                            

  Imbalance     Linear SVC +          0.826        0.769        0.893        0.980
  Handling      SMOTE                                                 

  Ensemble      **Stacking**      **0.850**        0.769        0.950        0.983
  Learning                                                            

  Frozen        DistilBERT +          0.773        0.688        0.881        0.982
  Transformer   XGBoost                                               
  --------------------------------------------------------------------------------

## Final Result

The **Stacking Ensemble** achieved the best overall F1-score among the
evaluated approaches:

**F1-score = 0.850**

It provided the strongest balance between Precision and Recall in the
experiments.

**Linear SVC + SMOTE** also performed strongly, reaching an F1-score of
**0.826**, while providing a simpler individual-model alternative.

## Repository Contents

``` text
fake_job_detection/
├── README.md
├── requirements.txt
├── Fake_Job_Detection_project.ipynb
├── Fake_Job_Posting_Detection_Report_Final.pdf
└── Fake_Job_Detection_Presentation.pptx
```

## Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Imbalanced-learn
-   XGBoost
-   NLTK / NLP preprocessing
-   TF-IDF
-   PyTorch
-   Hugging Face Transformers
-   DistilBERT
-   Matplotlib
-   Seaborn

## How to Run

Install the required packages:

``` bash
pip install -r requirements.txt
```

Then open:

``` text
Fake_Job_Detection_project.ipynb
```

and run the notebook cells in order.

## Notes

This repository contains the project notebook, report, and presentation.
The original dataset is not included.

The reported metrics are based on the project's held-out test set and
should be interpreted in the context of the dataset and its class
imbalance.

## Future Work

-   Evaluate additional Transformer architectures such as BERT or
    RoBERTa
-   Perform deeper error analysis
-   Optimize the classification threshold for deployment
-   Evaluate the model on external datasets to test generalization
-   Build and evaluate a deployment application
