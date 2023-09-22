import pandas as pd

# Load the Excel file into a DataFrame
data_df = pd.read_excel("大学生游戏生活情况调查问卷.xlsx")

#去除序号以及MBTI的的问题
# Drop the specified columns
data_df_cleaned = data_df.drop(columns=['序号', '4、您是否清楚您的MBTI人格？（该题不是必填，如果您愿意可以通过以下网址测试http://https://www.16personalities.com/）'])

# 获取分类值
answer_distribution = data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？'].value_counts()

# # 计算相关性
# correlations = data_df_cleaned.corrwith(data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？'])
#
# # Sort the correlations in descending order
# sorted_correlations = correlations.abs().sort_values(ascending=False)

#获取9.10.11.13的多维向量
import matplotlib.pyplot as plt

# List of questions with multi-features to be fused
questions_to_fuse = {
    '9': [col for col in data_df_cleaned.columns if col.startswith('9、')],
    '10': [col for col in data_df_cleaned.columns if col.startswith('10、')],
    '11': [col for col in data_df_cleaned.columns if col.startswith('11、')],
    '13': [col for col in data_df_cleaned.columns if col.startswith('13、')]
}



from sklearn.decomposition import PCA

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Create a dictionary to store PCA results for each question
pca_results = {}

# Apply PCA for each question's features and store results
for key, features in questions_to_fuse.items():
    pca_results[key] = pca.fit_transform(data_df_cleaned[features])

# Plot the PCA results with red-green color scheme and legends
plt.figure(figsize=(15, 10))
colors = {1: 'red', 2: 'green'}
labels = {1: 'yes', 2: 'no'}
#提取标题
questions_titles = {
    '9': 'Question 9',
    '10': 'Question 10',
    '11': 'Question 11',
    '13': 'Question 13'
}
#去除问题13中值空缺的位置
mask_valid_13 = (data_df_cleaned[questions_to_fuse['13']] != -3).any(axis=1)
pca_results['13'] = pca_results['13'][mask_valid_13]


#画出决策边界
from sklearn.linear_model import LogisticRegression
import numpy as np

logreg = LogisticRegression()


def plot_decision_boundary(X, y, model, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)


# Plot the PCA results with decision boundary
plt.figure(figsize=(15, 10))

for idx, (key, value) in enumerate(pca_results.items(), 1):
    ax = plt.subplot(2, 2, idx)
    target_values = data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？'][mask_valid_13] if key == '13' else \
    data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？']

    # Train logistic regression on PCA results
    logreg.fit(value, target_values)

    # Plot decision boundary
    plot_decision_boundary(value, target_values, logreg, ax)

    for label in [1, 2]:
        mask = target_values == label
        ax.scatter(value[mask, 0], value[mask, 1], c=colors[label], label=labels[label], edgecolors='k', alpha=0.5)
    ax.set_title(questions_titles[key])
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()

plt.tight_layout()
plt.show()

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Initialize LDA
lda = LDA()

# Create a dictionary to store LDA results for each question
lda_results = {}

# Apply LDA for each question's features and store results
for key, features in questions_to_fuse.items():
    lda_results[key] = lda.fit_transform(data_df_cleaned[features], data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？'])

plt.figure(figsize=(15, 10))

for idx, (key, value) in enumerate(lda_results.items(), 1):
    ax = plt.subplot(2, 2, idx)
    target_values = data_df_cleaned['17、您有没有玩过（或者愿意去玩）米哈游的游戏（崩坏学院、原神、星穹铁道）？']
    for label in [1, 2]:
        mask = target_values == label
        ax.scatter(value[mask, 0], np.zeros_like(value[mask, 0]), c=colors[label], label=labels[label], alpha=0.5)
    ax.set_title(questions_titles[key])
    ax.set_xlabel("LDA Component")
    ax.set_ylabel("Constant (LDA produces 1D output)")
    ax.legend()

plt.tight_layout()
plt.show()
