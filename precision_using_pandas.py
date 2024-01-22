import pandas as pd

# Assuming you have a DataFrame named df with 'ground_truth' and 'predicted' columns
# Replace df with your actual DataFrame name

# Example DataFrame
data = {'ground_truth': [1, 2, 1, 1, 2, 3],
        'predicted': [1, 2, 1, 3, 2, 3]}
df = pd.DataFrame(data)

# Function to calculate precision for a specific class
def calculate_precision(class_label):
    true_positives = ((df['ground_truth'] == class_label) & (df['predicted'] == class_label)).sum()
    false_positives = ((df['ground_truth'] != class_label) & (df['predicted'] == class_label)).sum()
    
    if true_positives + false_positives == 0:
        return 0  # To handle the case where denominator is zero
    
    precision = true_positives / (true_positives + false_positives)
    return precision

# Get unique class labels
unique_classes = df['ground_truth'].unique()

# Calculate precision for each class
precision_per_class = {class_label: calculate_precision(class_label) for class_label in unique_classes}

# Calculate Macro Precision (average precision across all classes)
macro_precision = sum(precision_per_class.values()) / len(precision_per_class)

print(f'Precision per class: {precision_per_class}')
print(f'Macro Precision: {macro_precision}')
