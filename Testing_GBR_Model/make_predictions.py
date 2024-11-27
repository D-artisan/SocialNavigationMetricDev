import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the extracted features
new_features_df = pd.read_csv('new_features_dataset.csv')

# Load the trained model and scaler
model = joblib.load('./training_output/trajectory_quality_model.pkl')
scaler = joblib.load('./training_output/scaler.pkl')

# Prepare the feature matrix
feature_columns = [
    'success', 'collision_occurred', 'wall_collisions', 'agent_collisions',
    'human_collisions', 'timeout_occurred', 'failure_to_progress',
    'stalled_time', 'time_to_goal', 'path_length',
    'success_weighted_path_length', 'v_min', 'v_avg', 'v_max', 'a_min',
    'a_avg', 'a_max', 'j_min', 'j_avg', 'j_max', 'cd_min', 'cd_avg',
    'space_compliance', 'min_distance_to_human', 'min_time_to_collision',
    'aggregated_time'
]

X_new = new_features_df[feature_columns]

# Handle missing values if necessary
X_new = X_new.fillna(0)

# Scale the features
X_new_scaled = scaler.transform(X_new)

# Predict quality scores
y_pred = model.predict(X_new_scaled)
y_pred = np.clip(y_pred, 0, 1)  # Ensure predictions are between 0 and 1

# Add predictions to the DataFrame
new_features_df['predicted_quality_score'] = y_pred

# Save the predictions
new_features_df.to_csv('new_features_with_predictions.csv', index=False)

print("Predictions saved to 'new_features_with_predictions.csv'.")

# Compute summary statistics
mean_score = np.mean(y_pred)
median_score = np.median(y_pred)
std_score = np.std(y_pred)
min_score = np.min(y_pred)
max_score = np.max(y_pred)
range_score = max_score - min_score

print("\nSummary Statistics of Predicted Quality Scores:")
print(f"Number of Trajectories: {len(y_pred)}")
print(f"Mean Quality Score: {mean_score:.4f}")
print(f"Median Quality Score: {median_score:.4f}")
print(f"Standard Deviation: {std_score:.4f}")
print(f"Minimum Quality Score: {min_score:.4f}")
print(f"Maximum Quality Score: {max_score:.4f}")
print(f"Range of Quality Scores: {range_score:.4f}")

# Optionally, save the summary statistics to a text file
with open('quality_score_summary.txt', 'w') as f:
    f.write("Summary Statistics of Predicted Quality Scores:\n")
    f.write(f"Number of Trajectories: {len(y_pred)}\n")
    f.write(f"Mean Quality Score: {mean_score:.4f}\n")
    f.write(f"Median Quality Score: {median_score:.4f}\n")
    f.write(f"Standard Deviation: {std_score:.4f}\n")
    f.write(f"Minimum Quality Score: {min_score:.4f}\n")
    f.write(f"Maximum Quality Score: {max_score:.4f}\n")
    f.write(f"Range of Quality Scores: {range_score:.4f}\n")

# Optionally, create a histogram of the predicted quality scores
plt.figure(figsize=(8, 6))
plt.hist(y_pred, bins=20, edgecolor='k', alpha=0.7)
plt.title('Distribution of Predicted Quality Scores')
plt.xlabel('Predicted Quality Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('quality_score_histogram.png')
plt.show()
