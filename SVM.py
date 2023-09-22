import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Excel file into a DataFrame
data_df = pd.read_excel("大学生游戏生活情况调查问卷.xlsx")

#去除不能量化的部分
cols_to_remove = ["序号", "提交答卷时间", "所用时间", "来源", "来源详情", "来自IP", "4、您是否清楚您的MBTI人格？（该题不是必填，如果您愿意可以通过以下网址测试http://https://www.16personalities.com/）"]

# Drop the specified columns
data_df_cleaned = data_df.drop(columns=cols_to_remove)

# Display the first few rows of the cleaned dataset to confirm
data_df_cleaned.head()

# Extract features and target variable
X_cleaned = data_df_cleaned.drop('17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？', axis=1)
y_cleaned = data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？']

X_cleaned.to_csv("X_train_data.csv", index=False)
y_cleaned.to_csv("y_train_data.csv", index=False)
# Train an SVM classifier
svm = SVC(kernel='linear', C=1)

# Train the SVM on the entire cleaned dataset
svm.fit(X_cleaned, y_cleaned)

# Predict on the entire cleaned dataset
y_pred_all_cleaned = svm.predict(X_cleaned)

# Evaluate performance on the entire dataset
accuracy_all_cleaned = accuracy_score(y_cleaned, y_pred_all_cleaned)
class_report_all_cleaned = classification_report(y_cleaned, y_pred_all_cleaned)
conf_matrix_all_cleaned = confusion_matrix(y_cleaned, y_pred_all_cleaned)

print(accuracy_all_cleaned, class_report_all_cleaned, conf_matrix_all_cleaned)

#交叉验证
from sklearn.model_selection import cross_val_score

# Use 5-fold cross-validation to evaluate SVM's performance
cross_val_scores = cross_val_score(svm, X_cleaned, y_cleaned, cv=5, scoring='accuracy')

# Calculate mean and standard deviation of cross-validation scores
cross_val_mean = cross_val_scores.mean()
cross_val_std = cross_val_scores.std()

print(cross_val_mean, cross_val_std)

#最高正确率
max_cross_val_accuracy = cross_val_scores.max()
print(max_cross_val_accuracy)

#输出交叉验证的混淆矩阵
from sklearn.model_selection import cross_val_predict

# Use cross_val_predict to get the predicted values for each fold
y_pred_cross_val = cross_val_predict(svm, X_cleaned, y_cleaned, cv=5)

# Compute the confusion matrix for the cross-validated predictions
conf_matrix_cross_val = confusion_matrix(y_cleaned, y_pred_cross_val)

print(conf_matrix_cross_val)

#输出正确率最高的SVM的混淆矩阵，并且保存模型
from sklearn.model_selection import KFold
import joblib

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

max_accuracy = 0
best_conf_matrix = None
best_model = None

# Iterate over each fold
for train_index, test_index in kf.split(X_cleaned):
    X_train_fold, X_test_fold = X_cleaned.iloc[train_index], X_cleaned.iloc[test_index]
    y_train_fold, y_test_fold = y_cleaned.iloc[train_index], y_cleaned.iloc[test_index]

    # Train the SVM on the fold's training data
    svm.fit(X_train_fold, y_train_fold)

    # Predict on the fold's test data
    y_pred_fold = svm.predict(X_test_fold)

    # Calculate accuracy
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)

    # If this fold's accuracy is the highest so far, store its confusion matrix
    if accuracy_fold > max_accuracy:
        max_accuracy = accuracy_fold
        best_conf_matrix = confusion_matrix(y_test_fold, y_pred_fold)
        best_model = svm

# #保存model
# model_path = "/best_svm_model.pkl"
# joblib.dump(best_model, model_path)

print(best_conf_matrix)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names = ["Yes", "No"]

plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('best_conf_matrix')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cross_val, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('conf_matrix_cross_val')
plt.show()
