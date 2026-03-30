from sklearn.linear_model import LogisticRegression
from preprocessing import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# 1. Load and Split Data
X_train, X_test, y_train, y_test, monthly_sales, train_dates, test_dates = load_data()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''
# 2. Initialize and Train
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# 3. Evaluate Performance
accuracy_score = classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy_score:.2f}")

In our project, we mainly use the "Order_Quantity" as our target variable for all regression models. 

If we use the target variable as "Order_Quantity" for logistic model, the accuracy score will only be 8% or 0.08. 
This happens because Order_Quantity is continuous numerical data, but the logistic regression is designed for classification tasks
where the output is categorical. As the results, the model incorrectly treated each unique sales value as a separate class,
leading to poor performance and convergence warnings.
To fix this issue, I should either switch to different regression models such as a decision tree or linear regression, 
or convert the target variable into categories (such as low, medium, high sales) before applying logistic regression. 

'''

#Let covert the Order_Quantity into 3 categories Low, Medium, and High using quantile-based bin

# Categorize data
# we find the cut-off point only using the training data (y_train)

q1 = y_train.quantile(0.33) # 33rd percentile
q2 = y_train.quantile(0.66) # 66th percentile

# We define our bins: Anything from 0 to q1 is Low, q1 to q2 is Medium, q2 to 999999 is high
max_order_quantity = monthly_sales["Order_Quantity"].max()
bin_thresholds = [-1, q1, q2, max_order_quantity]
bin_names = ["Low", "Medium", "High"]


# Convert the numerical numbers into category names
y_train_names = pd.cut(y_train, bins=bin_thresholds, labels=bin_names)
y_test_names = pd.cut(y_test, bins=bin_thresholds, labels=bin_names)

# Encode target - convert "low, medium, high" to 0,1,2
labelEncoder = LabelEncoder()
y_train_encoded = labelEncoder.fit_transform(y_train_names)
y_test_encoded = labelEncoder.transform(y_test_names)

# Initialize and train the model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train_encoded)

#Predict
prediction_encoded = classifier.predict(X_test)  # numerical (0-2)

#Convert prediction back to label
prediction_label = labelEncoder.inverse_transform(prediction_encoded)
y_test_label = labelEncoder.inverse_transform(y_test_encoded)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, prediction_encoded))

print("\nClassification Report:")
print(classification_report(y_test_encoded, prediction_encoded))

results = pd.DataFrame({
    "Date": monthly_sales.loc[test_dates, "Date"],
    "Actual_Label": y_test_label,
    "Predicted_Label": prediction_label
})
mapping = {"Low": 0, "Medium": 1, "High": 2}

results["Actual_Level"] = results["Actual_Label"].map(mapping)
results["Predicted_Level"] = results["Predicted_Label"].map(mapping)

# Aggregate by Date
plot_values = results.groupby("Date")[["Actual_Level", "Predicted_Level"]].mean().reset_index()

#  Plot Sales Level

plt.plot(plot_values["Date"], plot_values["Actual_Level"], label="Actual_Level", color="blue")
plt.plot(plot_values["Date"], plot_values["Predicted_Level"], label="Predicted_Level", color="red")
plt.yticks([0, 1, 2], ["Low", "Medium", "High"])

plt.xlabel("Date")
plt.ylabel("Sales Level (0=Low,1=Med,2=High)")
plt.title("Actual vs Predicted Sales Level")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Accuracy Over Time
# -------------------------------
#
# results["Correct"] = results["Actual"] == results["Predicted"]
#
# accuracy_by_date = results.groupby("Date")["Correct"].mean()
#
# plt.figure()
# plt.plot(accuracy_by_date.index, accuracy_by_date.values)
#
# plt.title("Model Accuracy Over Time")
# plt.xlabel("Date")
# plt.ylabel("Accuracy")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()