import pandas as pd
import holidays

# Sample DataFrame
data = {
    'date': ['2024-01-01', '2024-07-04', '2024-12-25', '2024-11-28', '2024-03-15'],
    'event': ['New Year Party', 'Independence Day Celebration', 'Christmas', 'Thanksgiving', 'Regular Day']
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Get US holidays for 2024
us_holidays = holidays.UnitedStates(years=[2024])

# Create a DataFrame from the holidays
holidays_df = pd.DataFrame(list(us_holidays.items()), columns=['date', 'holiday'])
holidays_df['date'] = pd.to_datetime(holidays_df['date'])

# Merge the two DataFrames on the date column
merged_df = pd.merge(df, holidays_df, on='date', how='left')

# Fill NaN values in the holiday column with 'No Holiday'
merged_df['holiday'].fillna('No Holiday', inplace=True)

print(merged_df)
