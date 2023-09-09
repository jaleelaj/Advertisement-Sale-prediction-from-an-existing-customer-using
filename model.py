import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load your dataset (replace 'DigitalAd_dataset.csv' with your actual dataset file)
df = pd.read_csv('DigitalAd_dataset.csv')

# Remove duplicates and reset index
df = df.drop_duplicates().reset_index(drop=True)

# Assuming your target variable is the last column, you can split the data
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# Train a Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)
model.fit(xtrain, ytrain)

# Make predictions on the test set
pred = model.predict(xtest)

# Calculate the accuracy of the model
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(ytest, pred)
print("Accuracy of the Model: {:.2f}%".format(accuracy * 100))

# Create a confusion matrix and display it using seaborn
conf_matrix = confusion_matrix(ytest, pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save the trained model to a file using pickle
pickle.dump(model, open('model.pkl', 'wb'))
