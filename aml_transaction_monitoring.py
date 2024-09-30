import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore

# Load the transaction data into a pandas dataframe
df = pd.read_csv("assets/transactions.csv")
# scaler = StandardScaler()
# df["amount"] = scaler.fit_transform(df[["amount"]])

# # Use KMeans to cluster the transactions into two groups
# kmeans = KMeans(n_clusters=2)
# df["cluster"] = kmeans.fit_predict(df[["amount"]])

# # Use RandomForestClassifier to build a fraud detection model
# model = RandomForestClassifier()
# model.fit(df[["amount", "cluster"]], df["fraud"])

# # Predict the fraud probability for new transactions
# new_transactions = [[1.5, 0], [0.5, 1], [-0.5, 1]]
# print(model.predict_proba(new_transactions))
# ------
# # Check columns
# print(df.columns)

# # Convert categorical 'type' column to numeric using LabelEncoder
# labelencoder = LabelEncoder()
# df['type'] = labelencoder.fit_transform(df['type'])

# # Apply scaling only to numeric columns
# scaler = StandardScaler()

# # Select only numeric columns for scaling (excluding any other non-numeric columns)
# df_numeric = df.select_dtypes(include=[float, int])

# # Scale the numeric columns
# df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# # Optionally, concatenate the scaled data back with non-numeric columns like 'type'
# df_final = df_scaled

# print(df_final)


# # Split the data into training and testing sets

# # Ensure the 'fraud' column (target) is categorical if it's continuous
# if df['fraud'].dtype in ['float64', 'int64']:
#     # If it's continuous, binarize it or label it as necessary (this is just an example)
#     df['fraud'] = (df['fraud'] > 0.5).astype(int)  # This is just an example for binary classification

# print(df['fraud'].unique())  # This will show the unique values in the 'fraud' column

# # If the target values are strings (e.g., "fraud", "not fraud"), encode them
# print(df['fraud'].unique()) 
# print(df_final)
# # Split the dataset into features (X) and target (y)
# X = df_final.drop(df['fraud'], axis=1)
# y = df_final[df['fraud']]
# print(X)
# print(y)
# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(y_train.dtype)
# print(y_train.unique())
# # Train the RandomForestClassifier model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# # Make predictions on new transactions (ensure the new data is scaled similarly)
# new_transactions = [[0.5, 0, 1, 0], [1.5, 0, 0, 1]]  # Example values
# new_transactions_scaled = scaler.transform(new_transactions)
# print(model.predict(new_transactions_scaled))


# Handle missing values
# Convert categorical 'type' column to numeric using LabelEncoder
v