# Stroke-Prediction
Practice machine learning pipeline using imbalanced dataset. For a small dataset of 992 samples, you could get high accuracy by predicting all cases as negative, but you won't detect any potential stroke victims. In this case, I used SMOTE to oversample the minority class (stroke) to get a more balanced dataset. Apply various models to the dataset (logistic regression does fairly well here) and attempt hyperparamter tuning.

Three metrics were used here: AUC score (how well does this detect true positives), Confusion matrix (detailed rundown), and  F1 score (mix of accuracy and precision). Choosing the right metric is important especially with imbalanced datasets. 

Frameworks used: pandas, sklearn, imblearn

Link to dataset used: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

[If I have time, I'll polish this page up with a proper report]

Flowchart for binary classification metric
![Flowchart for binary classification metric](https://machinelearningmastery.com/wp-content/uploads/2019/12/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)
Source: https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
