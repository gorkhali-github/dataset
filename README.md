📊 Machine Learning Datasets

This repository presents a comprehensive experimental comparison of different data preprocessing techniques and machine learning models across multiple datasets.

The project evaluates:

Missing value strategies

Encoding techniques

Label imbalance handling

Scaling methods

Classification models

Regression models

Each dataset is tested systematically, and the best-performing approach is selected based on evaluation metrics.

🔹 1️⃣ Missing Value Handling

Different imputation strategies were tested:

Drop rows

Mean

Median

KNN

Forward Fill (ffill)

Backward Fill (bfill)

📌 Best Strategies Per Dataset
Dataset	Best Strategy	Reason
data_1	✅ Mean	Highest R² + Lowest MSE
data_2	✅ Median	Best R², MSE, MAE
data_3	✅ KNN	Clearly best performance margin
data_4	✅ Drop	Removing noisy rows preserved structure
🔎 Key Insight

Mean works well for normally distributed data.

Median is robust to outliers.

KNN captures complex feature relationships.

Dropping rows is effective when imputation distorts data heavily.

🔹 2️⃣ Encoding Strategies

Tested:

Ordinal Encoding

One-Hot Encoding

📌 Best Encoding Per Dataset
Dataset	Best Encoding
data_1	✅ Ordinal
data_2	✅ One-Hot
🔎 Insight

If categorical variables have order → Ordinal works better

If categorical variables are nominal → One-Hot is superior

🔹 3️⃣ Label Distribution (Class Imbalance Handling)

Tested:

None

Under-sampling

Over-sampling

SMOTE

Class Weights

📌 Best Strategy Per Dataset
Dataset	Best Strategy
data_1	✅ None
data_2	✅ None
data_3	✅ None
data_4	✅ Weights
🔎 Insight

Data_1,2,3 → Balanced datasets → resampling hurts performance.

Data_4 → Imbalanced → Class weights improved Recall & F1 significantly.

🔹 4️⃣ Classification Models

Models Tested:

Logistic Regression

KNN

SVM

Decision Tree

📌 Best Model Per Dataset
Dataset	Best Model
data_4	✅ Logistic Regression
data_5	✅ KNN
data_6	✅ SVM
data_7	✅ Decision Tree
🔎 Insight

Logistic → Best for linear separability

KNN → Works well for clustered data

SVM → Strong for smooth nonlinear boundaries

Decision Tree → Best for rule-based splits

🔹 5️⃣ Regression Models

Models Tested:

Linear Regression

KNN Regressor

SVM Regressor

Decision Tree Regressor

📌 Best Regression Model
Dataset	Best Model
data_1	✅ Linear Regression
data_2	✅ Decision Tree
data_3	✅ SVM
🔎 Insight

Data_1 → Strong linear relationship

Data_2 → Highly nonlinear (tree-based splits dominate)

Data_3 → Smooth nonlinear boundary (SVM best)

🔹 6️⃣ Feature Scaling

Tested:

None

Min-Max Scaling

Standard Scaling

📌 Best Scaling Per Dataset
Dataset	Best Scaling
data_1	✅ Min-Max (or Standard)
data_2	✅ None
data_3	✅ Standard
🔎 Insight

Linear & SVM models → Sensitive to scaling

Tree-based models → Scale invariant

Always evaluate scaling impact per dataset

📈 Key Learnings

✔ No single preprocessing technique works best for all datasets.
✔ Data characteristics determine the optimal strategy.
✔ Model selection must align with dataset structure.
✔ Preprocessing decisions significantly impact performance.

🧠 Final Summary
Task	Key Finding
Missing Values	Strategy depends on distribution & structure
Encoding	Use ordinal for ordered categories, one-hot for nominal
Imbalance	Use weights only when necessary
Scaling	Required for linear & distance-based models
Classification	Model performance depends on decision boundary shape
Regression	Choose model based on linear vs nonlinear behavior
🚀 Conclusion

This project demonstrates the importance of:

Careful preprocessing

Comparative evaluation

Metric-based model selection

Understanding dataset structure before choosing algorithms