# Enhancing Maternal Health Risk Prediction: Comparing Original and Feature-Engineered Data Using ML Models and Meta-Models
## Project Overview: 
This research investigates the predictive performance of maternal health risk models using both original and feature-engineered data. The study compares the performance of individual traditional machine learning models, a Feedforward Neural Network (FNN), and ensemble models. The goal is to determine whether a meta-model trained on the combined predictions of these models can enhance predictive ability compared to using each model individually.
Furthermore, the research also examines the role of feature engineering and its influence on model performance and whether combining model predictions via ensemble methods or meta-models can improve predicting accuracy for maternal health risk prediction. This has practical implications for improving early intervention and care in maternal health.

## System specification. 
The experiment was conducted on a Windows 10 Pro (Version 10.0.19045) system with the following specifications:
- RAM: 8 GB
- Processor: Intel Core i5, 2.4 GHz

## Execution Environment and Library versions used. 
The code was executed in a Jupyter Notebook environment using Python 3.11.4. The following library versions were used: scikit-learn 1.4.2, Keras 3.4.1, TensorFlow 2.16.1, Matplotlib 3.7.1, Pandas 2.1.1, and NumPy 1.24.3.

## Project Code
The code expects a dataset in CSV format. The preprocessing steps include:
- Handling missing values
- Encoding categorical features
- Separating the target feature from the input features
- Splitting the dataset into training and test sets (80:20 ratio)
- Standardizing the data using StandardScaler
Feature engineering was performed by creating a new feature called Mean Arterial Pressure (a combination of systolic and diastolic pressure values). A log transformation using Box-Cox was applied to correct skewed data.

## Models used
Random Forest (RF) and a Feedforward Neural Network (FNN) were selected as base models for the ensemble models. Both models were tuned to optimize performance:

Random Forest: Max depth of 16 and 100 estimators (n_estimators=100).
FNN: 3 hidden layers, using the softmax activation function. The model was trained using the Adam optimizer with a learning rate of 0.0001 over 400 epochs.

Meta-models were applied to the ensemble predictions:

Logistic Regression (LR) was used for the original data.
XGBoost (XGB) was used for feature-engineered data.

## Performance
The XGB meta-model trained on the ensemble predictions of the feature-engineered dataset demonstrated strong performance after hyperparameter tuning (max depth: 10, n_estimators: 50, learning_rate: 0.1). It achieved the following metrics:

Accuracy: 91%
Precision: 91%
Recall: 91%
F1 Score: 91%
AUC: 96%

## Limitations.
9.4% of the pregnant women in the dataset were aged 50 years and above. Since the data did not specify whether these pregnancies were naturally conceived or through Assisted Reproductive Technologies (ART), this affects the generalizability of the model's performance. Additionally, it was unclear whether the data entries were single-time records or averages over a period, and if the dataset captured full-term pregnancies.

The dataset size was also a limiting factor, as the small sample size hindered the models' ability to learn effectively, potentially impacting predictive performance.

## Future work.
Incorporate more features that captures medical and socio-economic history, explore data from other regions, validate the midel using external data, include time-series data to track changes in maternal health metrics over time. This could help in predicting risks earlier and more accurately, especially when dealing with IoT-generated data.
