# Import necessary libraries
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Data Preprocessing
file_path = 'trends_by_cat.csv'
df = pd.read_csv(file_path)
df = df[df["location"] == "Global"]

df["frequency"] = df["query"].map(df["query"].value_counts())
print(df)
#print(len(pd.unique(df["query"])))









# # Move category to last column
# i = list(df.columns)
# a, b = i.index("category"), i.index("query")
# i[b], i[a] = i[a], i[b]
# df = df[i]

# # Assuming the last column is the target variable (category) and the rest are features
# X = df["rank"]  # Features
# y = df["category"]  # Category

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create an SVM model
# svm_model = SVC(kernel='linear')

# # Train the model on the training data
# svm_model.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = svm_model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # Print a classification report for more detailed metrics
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
