# ML Assignment 2 - Bank Marketing Classification

## A. Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification models on a classification dataset obtained from a public repository.

For this assignment, the Bank Marketing dataset has been selected. The objective is to predict whether a customer will subscribe to a term deposit based on customer and marketing campaign information.

This is a binary classification problem where:

- `yes` indicates that the customer subscribed to a term deposit.
- `no` indicates that the customer did not subscribe to a term deposit.

The classification models implemented for this assignment are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN) Classifier
4. Naive Bayes Classifier
5. Random Forest Ensemble Model

The models are compared using the following evaluation metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The objective is to compare the performance of the implemented models and identify the overall best-performing model for the Bank Marketing classification problem.

---

## B. Dataset Description

### Dataset Name

Bank Marketing Dataset

### Dataset Source

UCI Machine Learning Repository

### Dataset URL

https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

### Dataset Description

The Bank Marketing dataset contains information collected during direct marketing campaigns conducted by a Portuguese banking institution.

The classification task is to predict whether a customer will subscribe to a term deposit.

The dataset satisfies the assignment requirements for a classification dataset with more than 500 instances and more than 12 input features.

### Target Variable

The target variable is:

`y`

The target variable contains two classes:

- `yes` - customer subscribed to a term deposit
- `no` - customer did not subscribe to a term deposit

### Dataset Loading

The dataset is imported programmatically from the public UCI URL:

https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

The dataset is therefore loaded from a public URL rather than being manually uploaded as the primary training dataset.

The held-out test data used for model evaluation is stored in:

`test_data.csv`

---

## C. GitHub Repository Link

GitHub Repository:

https://github.com/Shubhamgogavale3007/ml-assignment2

---

## D. Models Used

### 1. Logistic Regression

Logistic Regression was implemented as a binary classification model to predict whether a customer would subscribe to a term deposit.

### 2. Decision Tree Classifier

Decision Tree Classifier was implemented as a tree-based classification algorithm for predicting the target class from the input features.

### 3. K-Nearest Neighbor (kNN) Classifier

K-Nearest Neighbor was implemented as a distance-based classification algorithm that predicts the target class based on neighboring observations.

### 4. Naive Bayes Classifier

Naive Bayes was implemented as a probabilistic classification algorithm based on Bayes' theorem.

### 5. Random Forest Ensemble Model

Random Forest was implemented as an ensemble classification algorithm using multiple decision trees.

---

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.845737 | 0.907922 | 0.418244 | 0.814745 | 0.552741 | 0.509218 |
| Decision Tree | 0.805485 | 0.888263 | 0.361736 | 0.866730 | 0.510437 | 0.475227 |
| kNN | 0.901471 | 0.884169 | 0.670061 | 0.310964 | 0.424790 | 0.412302 |
| Naive Bayes | 0.854805 | 0.810095 | 0.405904 | 0.519849 | 0.455864 | 0.377358 |
| Random Forest (Ensemble) | 0.871724 | 0.924429 | 0.471477 | 0.796786 | 0.592410 | 0.547506 |

---

## Observations About Model Performance

### Logistic Regression

Logistic Regression achieved an Accuracy of 0.845737 and an AUC of 0.907922. It achieved a Recall of 0.814745, indicating good identification of positive cases. Its Precision was 0.418244, F1 Score was 0.552741, and MCC was 0.509218. Overall, Logistic Regression provided a strong baseline with good AUC and Recall performance.

### Decision Tree Classifier

Decision Tree achieved an Accuracy of 0.805485 and an AUC of 0.888263. It achieved the highest Recall among the implemented models at 0.866730, indicating strong ability to identify positive cases. However, its Precision was 0.361736, F1 Score was 0.510437, and MCC was 0.475227. Its Accuracy was the lowest among the implemented models.

### K-Nearest Neighbor (kNN) Classifier

kNN achieved the highest Accuracy among the implemented models at 0.901471. It also achieved the highest Precision at 0.670061. However, its Recall was only 0.310964, indicating that it missed a significant proportion of positive cases. Its F1 Score was 0.424790 and MCC was 0.412302. Therefore, although kNN achieved the highest Accuracy and Precision, it did not provide the best balance between Precision and Recall.

### Naive Bayes Classifier

Naive Bayes achieved an Accuracy of 0.854805 and an AUC of 0.810095. Its Precision was 0.405904, Recall was 0.519849, F1 Score was 0.455864, and MCC was 0.377358. Its AUC and MCC were lower than those of the other models, indicating comparatively weaker overall classification performance.

### Random Forest Ensemble Model

Random Forest achieved an Accuracy of 0.871724 and the highest AUC of 0.924429. It also achieved the highest F1 Score of 0.592410 and the highest MCC of 0.547506. Its Recall was 0.796786, which was also strong. Although its Accuracy was lower than kNN, Random Forest provided the strongest overall balance across the evaluation metrics.

---

## Overall Winner

### Random Forest (Ensemble)

Random Forest is selected as the overall winner for the Bank Marketing classification problem.

The model achieved:

- Highest AUC: 0.924429
- Highest F1 Score: 0.592410
- Highest MCC: 0.547506
- Recall: 0.796786
- Accuracy: 0.871724

Although kNN achieved the highest Accuracy and Precision, its Recall was considerably lower. Decision Tree achieved the highest Recall, but its Accuracy and Precision were lower.

Random Forest provided the strongest overall balance across AUC, Precision, Recall, F1 Score and MCC and is therefore selected as the overall best-performing model.