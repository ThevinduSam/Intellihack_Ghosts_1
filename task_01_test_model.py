import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tkinter import filedialog

# Load the saved Decision Tree model
clf = joblib.load('decision_tree_model.joblib')

# Open a file dialog to select the new dataset file
file_path = filedialog.askopenfilename(title="Crop_Dataset")

# Load the new dataset for prediction
new_data = pd.read_csv(file_path)

# Preprocess the new dataset (similar to how the original dataset was preprocessed)
X_new = new_data.drop('Label_Encoded', axis=1)
X_new = X_new.drop('Label', axis=1)
y_new = new_data['Label_Encoded']

# Split the new dataset into training and testing sets (40% of the data)
X_new, _, y_new, _ = train_test_split(X_new, y_new, test_size=0.99, random_state=42)

# Make predictions on the new dataset using the loaded model
y_pred_new = clf.predict(X_new)

# Calculate the accuracy of the model on the new dataset
accuracy_new = accuracy_score(y_new, y_pred_new)
print(f'Accuracy of Decision Tree model on the new dataset: {accuracy_new}')

# Plot the results for the new dataset
y_new = np.array(y_new)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_new)), y_new, label='test', color='blue', alpha=0.7)
plt.scatter(range(len(y_pred_new)), y_pred_new, label='predicted', color='red', alpha=0.5)
plt.grid()
plt.legend()
plt.title('Decision Tree on New Dataset')
plt.xlabel('n')
plt.ylabel('crop')
plt.show()