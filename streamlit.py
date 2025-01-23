import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace this with loading your actual dataset
df = pd.read_csv('student_depression.csv')

st.title('Model Performance App')

st.write("Dataset Preview:")
st.dataframe(df.head())

# List of columns to drop
columns_to_drop = [
    'Sleep Duration_2', 'City_Region_East India', 'Gender_Female', 'City_Region_Central India', 
    'City_Region_West India', 'City_Region_North India', 'Gender_Male', 'New_Degree_Post Graduate', 
    'New_Degree_Undergraduate', 'City_Region_South India', 'Family History of Mental Illness_Yes', 
    'Sleep Duration_1', 'New_Degree_Professional', 'Family History of Mental Illness_No', 
    'Dietary Habits_Moderate', 'Sleep Duration_3', 'Sleep Duration_0'
]

df = df.drop(columns=columns_to_drop)

# Define features (X) and target (y)
X = df.drop('Depression', axis=1).to_numpy()  # Features
y = df['Depression'].to_numpy()  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=7)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting Regressor model
GBR = GradientBoostingRegressor(
    n_estimators=3000, 
    min_samples_leaf=5, 
    max_features=0.3, 
    max_depth=4, 
    loss='huber', 
    learning_rate=0.01
)

# Train the model
GBR.fit(X_train_scaled, y_train)

# Predict using the trained model
y_pred = GBR.predict(X_test_scaled)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.subheader(f"Performance:")
st.write(f"MAE: {mae:.2f}")
st.write(f"R²: {r2:.2f}")

# Feature importance from the trained model
st.subheader("Feature Importance")
feature_importances = GBR.feature_importances_

# Create a DataFrame for feature importances
feature_labels = df.drop('Depression', axis=1).columns
feature_importances_df = pd.DataFrame({
    'Feature': feature_labels,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
st.pyplot()

# End the app with some text
st.markdown("Thank you for using the Model Performance App!")
