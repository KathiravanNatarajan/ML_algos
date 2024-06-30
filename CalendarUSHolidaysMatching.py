import pandas as pd
import holidays

def get_holidays(date_column):
    """
    This function takes a pandas Series (column) with dates as input
    and returns a Series with 1 for holidays and 0 for non-holidays based on the U.S. calendar.
    
    Parameters:
    date_column (pd.Series): A pandas Series containing dates.
    
    Returns:
    pd.Series: A pandas Series with binary values (1 for holidays and 0 for non-holidays).
    """
    # Create a US holidays object
    us_holidays = holidays.US()
    
    # Apply a function to check if each date is a holiday
    return date_column.apply(lambda x: 1 if x in us_holidays else 0)

# Example usage:
data = {
    'date': pd.to_datetime(['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04']),
    'temperature': [75, 78, 82, 85],
    'holiday': ['None', 'None', 'Independence Day', 'None'],  # Example: holiday names as strings
    'sales': [250, 280, 320, 400]  # Example: target variable
}

df = pd.DataFrame(data)

# Add the new 'is_holiday' column to the DataFrame using the get_holidays function
df['is_holiday'] = get_holidays(df['date'])

# Print the DataFrame to see the result
print(df)
