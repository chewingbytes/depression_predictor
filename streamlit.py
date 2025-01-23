import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Replace with your actual dataset path)
df = pd.read_csv('student_depression.csv')

st.title('Model Performance App')

st.write("Dataset Preview:")
st.dataframe(df.head())

# Data Preprocessing Steps
# Drop the 'id' column
df = df.drop(['id'], axis=1)

# Remove cities with fewer than 400 occurrences
cities_to_remove = df['City'].value_counts()[df['City'].value_counts() < 400]
df = df[~df['City'].isin(cities_to_remove.index)]
df['City'].value_counts()

# Assign city regions
north_india = ['Srinagar', 'Lucknow', 'Delhi', 'Ghaziabad', 'Faridabad', 'Jaipur', 'Patna', 'Meerut', 'Kanpur', 'Agra']
south_india = ['Hyderabad', 'Chennai', 'Visakhapatnam', 'Bangalore', 'Pune']
west_india = ['Vasai-Virar', 'Thane', 'Mumbai', 'Ahmedabad', 'Surat', 'Rajkot', 'Vadodara', 'Nashik']
east_india = ['Kolkata', 'Varanasi']
central_india = ['Bhopal', 'Indore', 'Nagpur']

df.loc[df['City'].isin(north_india), 'City_Region'] = 'North India'
df.loc[df['City'].isin(south_india), 'City_Region'] = 'South India'
df.loc[df['City'].isin(west_india), 'City_Region'] = 'West India'
df.loc[df['City'].isin(east_india), 'City_Region'] = 'East India'
df.loc[df['City'].isin(central_india), 'City_Region'] = 'Central India'

df['City_Region'].value_counts()

# Drop the 'City' column after assigning regions
df = df.drop(['City'], axis=1)

# Filter data where 'Age' <= 30
df = df.loc[df['Age'] <= 30]
df['Age'].value_counts()

# Remove rows where 'Sleep Duration' is 'Others'
df = df.loc[df['Sleep Duration'] != 'Others']
df['Sleep Duration'].value_counts()

# Convert sleep duration values to numeric
df.loc[df['Sleep Duration'] == 'Less than 5 hours', 'Sleep Duration'] = 0
df.loc[df['Sleep Duration'] == '5-6 hours', 'Sleep Duration'] = 1
df.loc[df['Sleep Duration'] == '7-8 hours', 'Sleep Duration'] = 2
df.loc[df['Sleep Duration'] == 'More than 8 hours', 'Sleep Duration'] = 3
df['Sleep Duration'].value_counts()

# Remove rows where 'Dietary Habits' is 'Others'
df = df.loc[df['Dietary Habits'] != 'Others']
df['Dietary Habits'].value_counts()

# Assign degree categories
undergraduate = ['Class 12', 'B.Ed', 'B.Arch', 'B.Com', 'BCA', 'B.Tech', 'BBA', 'BSc']
postgraduate = ['MSc', 'MCA', 'M.Tech', 'MCom', 'MA', 'MBA']
professional = ['MBBS', 'PhD', 'MD', 'LLB']

df.loc[df['Degree'].isin(undergraduate), 'New_Degree'] = 'Undergraduate'
df.loc[df['Degree'].isin(postgraduate), 'New_Degree'] = 'Post Graduate'
df.loc[df['Degree'].isin(professional), 'New_Degree'] = 'Professional'

# Remove rows where 'Degree' is 'Others'
df = df.loc[df['Degree'] != 'Others']

# One-hot encode categorical columns
features_df = pd.get_dummies(df, columns=["Gender", "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness", "New_Degree", "City_Region"])

# Remove rows with missing values
df_cleaned = features_df.dropna()

st.write("Missing Value Counts After Removal:")
st.write(df_cleaned.isnull().sum())

# Convert boolean columns to int
df_cleaned = df_cleaned.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

# Split features and target variable
X = df_cleaned.drop('Depression', axis=1)  # Features
y = df_cleaned['Depression']  # Target variable

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

# Get feature importances
feature_importances = GBR.feature_importances_

# Create a DataFrame for feature importances
feature_labels = X.columns
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
