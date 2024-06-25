import pandas as pd

# Load the data
data = pd.read_csv('-----.csv')

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

# Display the first few rows of the transformed dataset
print(melted_data.head())
