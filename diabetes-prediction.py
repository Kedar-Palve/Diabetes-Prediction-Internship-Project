import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the training dataset
train_df = pd.read_csv('train.csv')

# Preprocess the data
train_df.fillna(train_df.median(), inplace=True)
train_df.drop('ID', axis=1, inplace=True)
X_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Split the data into a training set and a validation set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Load the test dataset
test_df = pd.read_csv('diabetes_test.csv')

# Preprocess the data
test_df.fillna(test_df.median(), inplace=True)
test_ids = test_df['ID']
test_df.drop('ID', axis=1, inplace=True)
X_test = scaler.transform(test_df)

# Use the model to predict the outcomes for the test set
test_preds = clf.predict(X_test)

# Save the predicted outcomes to a CSV file
pred_df = pd.DataFrame({'ID': test_ids, 'predicted': test_preds})
pred_df.to_csv('predictions.csv', index=False)