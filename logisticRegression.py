from sklearn.linear_model import LogisticRegression
from preprocessing import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# 1. Load and Split Data
X_train, X_test, y_train, y_test, monthly_sales, train_dates, test_dates = load_data()


# 2. Initialize and Train
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# 3. Evaluate Performance
accuracy_score = classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy_score:.2f}")

'''
In our project, we mainly use the "Order_Quantity" as our target variable for all regression models. 

If we use the target variable as "Order_Quantity" for logistic model, the accuracy score will only be 8% or 0.08. 
This happens because Order_Quantity is continuous numerical data, but the logistic regression is designed for classification tasks
where the output is categorical. As the results, the model incorrectly treated each unique sales value as a separate class,
leading to poor performance and convergence warnings.
To fix this issue, I should either switch to different regression models such as a decision tree or linear regression, 
or convert the target variable into categories (such as low, medium, high sales) before applying logistic regression. 

'''

#Let covert the Order_Quantity into 3 categories Low, Medium, and High using quantile-based bin

monthly_sales["Sales_Level"] = pd.qcut(
    monthly_sales["Order_Quantity"],
    q=3,
    labels=["Low", "Medium", "High"]
)

y = monthly_sales["Sales_Level"]

# Encode target
labelEncode = LabelEncoder()
targetVariable = labelEncode.fit_transform(monthly_sales["Sales_Level"])

# correct split using train_dates and test_dates
y_train_category = targetVariable[train_dates]
y_test_category = targetVariable[test_dates]
# Train model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train_category)

# Evaluate
accuracy_score = classifier.score(X_test, y_test_category)
print(f"Model Accuracy: {accuracy_score:.2f}")

# The accuracy improve from 0.08 to 0.69
y_prediction = classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test_category, y_prediction)
print("confustion matrix",conf_matrix)
print(classification_report(y_test_category, y_prediction, target_names=["Low","Medium","High"]))

# plot actual and predict values over time.
# Display results
# ------------------
mapping = {"Low": 0, "Medium": 1, "High": 2}
results = pd.DataFrame({
    "Date": monthly_sales.loc[test_dates, "Date"],
    "Actual": labelEncode.inverse_transform(y_test_category),
    "Predicted": labelEncode.inverse_transform(y_prediction)
})

results["Actual_num"] = results["Actual"].map(mapping)
results["Predicted_num"] = results["Predicted"].map(mapping)

plot_values = results.groupby("Date")[["Actual_num", "Predicted_num"]].mean().reset_index()

plt.plot(plot_values["Date"], plot_values["Actual_num"], label="Actual")
plt.plot(plot_values["Date"], plot_values["Predicted_num"], label="Predicted")

plt.xlabel("Date")
plt.ylabel("Sales Level (0=Low,1=Med,2=High)")
plt.title("Actual vs Predicted Sales Level")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

