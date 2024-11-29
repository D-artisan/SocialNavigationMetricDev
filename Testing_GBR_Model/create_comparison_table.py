import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions and features
df = pd.read_csv('new_features_with_predictions.csv')

# Create a new DataFrame containing only the necessary columns
comparison_df = df[['predicted_quality_score', 'time_to_goal', 'cd_avg']]

# Rename the columns to match the table headers
comparison_df.rename(columns={
    'predicted_quality_score': 'Learned Quality Score',
    'time_to_goal': 'Time to Goal (s)',
    'cd_avg': 'Average Distance to Humans (m)'
}, inplace=True)

# Add Trajectory identifiers
comparison_df['Trajectory'] = ['Trajectory {}'.format(i + 1) for i in range(len(comparison_df))]

# Reorder columns to desired order
comparison_df = comparison_df[['Trajectory', 'Learned Quality Score', 'Time to Goal (s)', 'Average Distance to Humans (m)']]

# Round numerical values for better presentation
comparison_df['Learned Quality Score'] = comparison_df['Learned Quality Score'].round(4)
comparison_df['Time to Goal (s)'] = comparison_df['Time to Goal (s)'].round(2)
comparison_df['Average Distance to Humans (m)'] = comparison_df['Average Distance to Humans (m)'].round(2)

# Preview the table
print("Comparison Table Preview:")
print(comparison_df)

# Export the table to a CSV file
comparison_df.to_csv('comparison_table.csv', index=False)

# Save the table as an image
def save_df_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))  # Adjust width and height dynamically
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Save as an image
save_df_as_image(comparison_df, 'comparison_table.png')

print("\nComparison table saved to 'comparison_table.csv'.")


