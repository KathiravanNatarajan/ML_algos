import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('-------.csv')

# Melt the dataframe
melted_data = pd.melt(
    data, 
    id_vars=['date'], 
    value_vars=[f'hour_{i}' for i in range(24)], 
    var_name='hour', 
    value_name='num_employees'
)

# Convert 'hour' to numeric
melted_data['hour'] = melted_data['hour'].str.extract('(\d+)').astype(int)

# Convert 'date' to datetime
melted_data['date'] = pd.to_datetime(melted_data['date'])

# Extract day of the week
melted_data['day_of_week'] = melted_data['date'].dt.dayofweek

# Display the first few rows of the transformed dataset
print(melted_data.head())

# Features: date, day_of_week, and hour
X = melted_data[['date', 'day_of_week', 'hour']]

# Convert 'date' to ordinal for regression purposes
X['date'] = X['date'].map(pd.Timestamp.toordinal)

# Target: number of employees
y = melted_data['num_employees']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X_test['hour'], y_test, color='black', label='Actual')
plt.scatter(X_test['hour'], y_pred, color='blue', label='Predicted')
plt.xlabel('Hour')
plt.ylabel('Number of Employees')
plt.legend()
plt.show()

# Predict the number of employees for a specific date, day of the week, and hour
future_date = pd.Timestamp('2024-06-30').toordinal()
future_day_of_week = pd.Timestamp('2024-06-30').dayofweek
hour = 4

# Create a DataFrame for prediction
prediction_df = pd.DataFrame([[future_date, future_day_of_week, hour]], columns=['date', 'day_of_week', 'hour'])
predicted_employees = model.predict(prediction_df)

print(f'Predicted number of employees for date 2024-06-30, day {future_day_of_week}, hour {hour}: {predicted_employees[0]}')
