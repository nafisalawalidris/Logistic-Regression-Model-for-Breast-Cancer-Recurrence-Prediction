#!/usr/bin/env python
# coding: utf-8

# # Import software libraries and load the dataset

# In[1]:


# Import required libraries
import sys                             # Read system parameters
import os                              # Interact with the operating system
import numpy as np                     # Work with multi-dimensional arrays and matrices
import pandas as pd                    # Manipulate and analyze data
import matplotlib                      # Create 2D charts
import scipy as sp                     # Perform scientific computing and advanced mathematics
import sklearn                         # Perform data mining and analysis
import seaborn as sb                   # Perform data visualization

# Summarize software libraries used
print('Libraries used in this project:')
print('- NumPy {}'.format(np.__version__))
print('- Pandas {}'.format(pd.__version__))
print('- Matplotlib {}'.format(matplotlib.__version__))
print('- SciPy {}'.format(sp.__version__))
print('- Scikit-learn {}'.format(sklearn.__version__))
print('- Python {}\n'.format(sys.version))

# Read the raw dataset based on data from https://archive.ics.uci.edu/ml/datasets/breast+cancer
print('------------------------------------')
print('Loading the dataset.')
PROJECT_ROOT_DIR = '.'
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'breast_cancer_data')
print('Data files in this project:', os.listdir(DATA_PATH) )
data_raw_file = os.path.join( DATA_PATH, 'breast-cancer.csv' )
data_raw = pd.read_csv( data_raw_file )
print('Loaded {} records from {}.\n'.format(len(data_raw), data_raw_file))


# # Get acquainted with the data structure and preview the records

# In[2]:


# Show the various features and their data types

# View the first five records

# Show the various features and their data types
print('Data structure and data types:')
print(data_raw.info())

# View the first five records
print('First five records:')
print(data_raw.head())


# # Examine descriptive statistics

# In[3]:


# Show descriptive statistics
print('Descriptive statistics:')
print(data_raw.describe())


# # Use histograms to visualize the distribution of various features

# In[6]:


# Show histograms for each attribute in the dataset

import matplotlib.pyplot as plt
import seaborn as sb

# Set a custom color palette
custom_palette = sb.color_palette("husl", len(data_raw.columns))

# Use Seaborn style
sb.set_style("whitegrid")

# Show histograms for each attribute in the dataset
plt.figure(figsize=(12, 8))
for i, feature in enumerate(data_raw.columns):
    plt.subplot(3, 4, i+1)
    plt.hist(data_raw[feature], bins=20, color=custom_palette[i], edgecolor='black', alpha=0.7)
    plt.title(feature)
plt.tight_layout()
plt.show()


# # Split the data into training and validation sets and labels

# In[17]:


from sklearn.model_selection import train_test_split

# Specify the column to be included in the label set ('recurrence')
label_column = 'recurrence'

# Specify columns to be included in the training and validation sets (all other columns)
training_columns = [col for col in data_raw.columns if col != label_column]

# Split the training set, validation set, and labels for both
X_train, X_valid, y_train, y_valid = train_test_split(data_raw[training_columns],
                                                      data_raw[label_column],
                                                      test_size=0.2,
                                                      random_state=42)

# Compare the number of rows and columns in the original data to the training and validation sets
print(f'Original dataset shape: {data_raw.shape}')
print(f'Training set shape: {X_train.shape}')
print(f'Validation set shape: {X_valid.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Validation labels shape: {y_valid.shape}')


# In[8]:


# Preview the training data
print("Training Data:")
print(X_train.head())


# In[9]:


# Preview the labels
print("Labels:")
print(y_train.head())


# # Build the model

# In[19]:


# Create a logistic regression model, and use the validation data and labels to score it.

from sklearn.linear_model import LogisticRegression
from time import time

log_reg = LogisticRegression(solver='sag', C=0.05, max_iter=10000)
start = time()
log_reg.fit(X_train, np.ravel(y_train))
end = time()
train_time = end - start

prediction = log_reg.predict(X_valid)

# Score using the validation data.
score = log_reg.score(X_valid, y_valid)

print('Score on validation set: {:.0f}%'.format(score * 100))
print('Training time: {:.2f} ms'.format(train_time * 1000))


# # Test the model

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Specify the column to be included in the label set ('recurrence')
label_column = 'recurrence'

# Specify columns to be included in the training and validation sets (all other columns)
training_columns = [col for col in data_raw.columns if col != label_column]

# Split the training set, validation set, and labels for both
X_train, X_val, y_train, y_val = train_test_split(data_raw[training_columns],
                                                  data_raw[label_column],
                                                  test_size=0.2,
                                                  random_state=42)

# Create a logistic regression model with specific parameters
log_reg = LogisticRegression(solver='sag', C=0.05, max_iter=10000)

# Fit the model to the training data and labels
log_reg.fit(X_train, np.ravel(y_train))

# Make predictions on the validation set
y_pred = log_reg.predict(X_val)

# Calculate the accuracy of the model's predictions on the validation set
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy on the validation set: {:.2f}%'.format(accuracy * 100))

# Add columns to a copy of the test data to compare predictions against actual values
results_log_reg = X_test_raw.copy()
results_log_reg['PredictedRecurrence'] = log_reg.predict(X_test)
results_log_reg['ProbRecurrence'] = np.round(log_reg.predict_proba(X_test)[:, 0] * 100, 2)
results_log_reg['ProbNoRecurrence'] = np.round(log_reg.predict_proba(X_test)[:, 1] * 100, 2)

# View examples of the predictions compared to actual recurrence
print('Examples of predicted recurrence compared to actual recurrence:')
print(results_log_reg[['PredictedRecurrence', 'ProbRecurrence', 'ProbNoRecurrence', 'recurrence']].head(10))


# In[ ]:




