Machine Learning Datasets

Machine Learning Preprocessing & Model Benchmark Study

This project presents a complete experimental comparison of data preprocessing techniques and machine learning models across multiple datasets.

The goal of this study is to determine:

✅ Best missing value handling strategy

✅ Best encoding technique

✅ Best class imbalance method

✅ Best scaling approach

✅ Best classification model

✅ Best regression model

All strategies were evaluated using proper performance metrics and cross-validation.

📌 Project Structure

The experiments are divided into:

Missing Value Handling

Encoding Techniques

Label Distribution Handling

Classification Models

Regression Models

Feature Scaling

Each dataset was tested independently.

🔹 1️⃣ Missing Value Handling
Strategies Tested

Drop rows

Mean imputation

Median imputation

KNN imputation

Forward fill (ffill)

Backward fill (bfill)

✅ Best Strategy Per Dataset
Dataset	Best Strategy
data_1	Mean
data_2	Median
data_3	KNN
data_4	Drop
🔎 Observations

Mean works well for normally distributed data.

Median is more robust to outliers.

KNN performs best when feature relationships are complex.

Drop works when imputation introduces heavy distortion.

🔹 2️⃣ Encoding Techniques
Strategies Tested

Ordinal Encoding

One-Hot Encoding

✅ Best Encoding
Dataset	Best Encoding
data_1	Ordinal
data_2	One-Hot
🔎 Observations

If categorical variables have order → Ordinal works better.

If categorical variables are nominal → One-Hot performs better.

🔹 3️⃣ Label Distribution Handling
Strategies Tested

None

Under-sampling

Over-sampling

SMOTE

Class Weights

✅ Best Strategy
Dataset	Best Strategy
data_1	None
data_2	None
data_3	None
data_4	Weights
🔎 Observations

Data_1, Data_2, Data_3 were not heavily imbalanced.

Data_4 was clearly imbalanced → Class weights improved Recall and F1-score significantly.

🔹 4️⃣ Classification Models
Models Tested

Logistic Regression

KNN

SVM

Decision Tree

✅ Best Classification Model
Dataset	Best Model
data_4	Logistic Regression
data_5	KNN
data_6	SVM
data_7	Decision Tree
🔎 Observations

Logistic → Best for linear separable data

KNN → Excellent for clustered data

SVM → Strong for smooth nonlinear boundaries

Decision Tree → Best for rule-based splits

🔹 5️⃣ Regression Models
Models Tested

Linear Regression

KNN Regressor

SVM Regressor

Decision Tree Regressor

✅ Best Regression Model
Dataset	Best Model
data_1	Linear Regression
data_2	Decision Tree
data_3	SVM
🔎 Observations

Data_1 shows strong linear structure.

Data_2 is highly nonlinear.

Data_3 has complex nonlinear smooth patterns.

🔹 6️⃣ Feature Scaling
Strategies Tested

None

Min-Max Scaling

Standard Scaling

✅ Best Scaling
Dataset	Best Scaling
data_1	Min-Max (or Standard)
data_2	None
data_3	Standard
🔎 Observations

Linear and SVM models are sensitive to scaling.

Tree-based models are scale invariant.

Always test scaling based on model type.

📊 Key Takeaways

✔ There is no universal best preprocessing method.
✔ Dataset characteristics determine optimal strategy.
✔ Model selection must match data structure.
✔ Preprocessing has a major impact on performance.
✔ Always evaluate using proper metrics.

🧠 Final Summary
Category	Key Insight
Missing Values	Strategy depends on distribution & data quality
Encoding	Choose based on ordinal vs nominal nature
Imbalance	Use weights only when dataset is skewed
Scaling	Required for linear & distance-based models
Model Choice	Depends on linear vs nonlinear structure
🚀 Conclusion

This project demonstrates the importance of:

Careful preprocessing

Systematic experimentation

Metric-based comparison

Understanding dataset behavior before model selection
