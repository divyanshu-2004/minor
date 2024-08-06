import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib



# Load data
data = pd.read_csv('complex_ad_clicks.csv')

# Separate features and target variable
X = data.drop(['user_id', 'clicked', 'timestamp'], axis=1)
y = data['clicked']

# Define categorical and numerical features
categorical_features = ['gender', 'ad_type', 'device_type', 'browser', 'location']
numerical_features = ['age', 'previous_clicks']

# Create a preprocessor with one-hot encoding for categorical features and scaling for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Create a pipeline with the preprocessor and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)





# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')


# Save the model
joblib.dump(model, 'ad_click_model.pkl')


# Load the model
model = joblib.load('ad_click_model.pkl')

# Assume you have new data in a DataFrame new_data
new_data = pd.read_csv('complex_ad_clicks.csv')
new_data = new_data.drop(['user_id', 'timestamp'], axis=1)

# Preprocess the new data
new_data_preprocessed = model.named_steps['preprocessor'].transform(new_data)

# Make predictions
new_predictions = model.named_steps['classifier'].predict(new_data_preprocessed)


