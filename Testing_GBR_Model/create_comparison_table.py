import pandas as pd

# Load the predictions and features
df = pd.read_csv('new_features_with_predictions.csv')

# Create a new DataFrame containing only the necessary columns
comparison_df = df[['predicted_quality_score', 'collision_occurred', 'time_to_goal', 'cd_avg']]

# Rename the columns to match the table headers
comparison_df.rename(columns={
    'predicted_quality_score': 'Learned Quality Score',
    'collision_occurred': 'Collision Occurred',
    'time_to_goal': 'Time to Goal (s)',
    'cd_avg': 'Average Distance to Humans (m)'
}, inplace=True)

# Convert Collision Occurred from 0/1 to 'No'/'Yes'
comparison_df['Collision Occurred'] = comparison_df['Collision Occurred'].map({0: 'No', 1: 'Yes'})

# Add Trajectory identifiers
comparison_df['Trajectory'] = ['Trajectory {}'.format(i + 1) for i in range(len(comparison_df))]

# Reorder columns to desired order
comparison_df = comparison_df[['Trajectory', 'Learned Quality Score', 'Collision Occurred', 'Time to Goal (s)', 'Average Distance to Humans (m)']]

# Round numerical values for better presentation
comparison_df['Learned Quality Score'] = comparison_df['Learned Quality Score'].round(4)
comparison_df['Time to Goal (s)'] = comparison_df['Time to Goal (s)'].round(2)
comparison_df['Average Distance to Humans (m)'] = comparison_df['Average Distance to Humans (m)'].round(2)

# Preview the table
print("Comparison Table Preview:")
print(comparison_df.head())

# Export the table to a CSV file
comparison_df.to_csv('comparison_table.csv', index=False)

print("\nComparison table saved to 'comparison_table.csv'.")
