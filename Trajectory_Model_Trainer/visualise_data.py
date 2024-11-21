import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Directories and files
DATA_DIR = './trajectory_dataset/'
LABELS_FILE = 'labels.json'
MODEL_FILE = 'trajectory_quality_model.pkl'
SCALER_FILE = 'scaler.pkl'

def load_trajectory_data(data_dir):
    """
    Loads trajectory data from JSON files in the specified directory.
    """
    trajectories = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                trajectories.append({
                    'filename': filename,
                    'data': data
                })
    return trajectories

def load_labels(labels_file):
    """
    Loads manual labels from the specified JSON file.
    """
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    return labels

def extract_features(trajectory):
    """
    Extracts features from a single trajectory.
    """
    data = trajectory['data']
    sequence = data.get('sequence', [])
    features = {}
    
    # Initialise variables
    num_collisions = 0
    wall_collisions = 0
    agent_collisions = 0
    human_collisions = 0
    timeout_occurred = 0
    failure_to_progress = 0
    stalled_time = 0.0
    path_length = 0.0
    velocities = []
    accelerations = []
    jerks = []
    clearing_distances = []
    min_distance_to_human = float('inf')
    min_time_to_collision = float('inf')
    timestamps = []
    
    # Flags from manual labels
    manual_label = trajectory.get('manual_label', {})
    success = 1 if manual_label.get('goal_reached', False) else 0
    collision_occurred = manual_label.get('collision_occurred', False)
    acceptable_social_nav = manual_label.get('acceptable_social_nav', None)
    
    # Process the sequence
    prev_time = None
    prev_position = None
    prev_velocity = None
    prev_acceleration = None
    for obs in sequence:
        timestamp = obs.get('timestamp', None)
        if timestamp is not None:
            timestamps.append(timestamp)
            if prev_time is not None:
                dt = timestamp - prev_time
                # Update stalled time
                if dt > 0:
                    speed = np.hypot(obs['robot'].get('speed_x', 0.0), obs['robot'].get('speed_y', 0.0))
                    if speed < 0.01:  # Threshold for being considered stalled
                        stalled_time += dt
            prev_time = timestamp
        
        # Positions
        current_position = np.array([obs['robot']['x'], obs['robot']['y']])
        if prev_position is not None:
            distance = np.linalg.norm(current_position - prev_position)
            path_length += distance
        prev_position = current_position
        
        # Velocities
        speed_x = obs['robot'].get('speed_x', 0.0)
        speed_y = obs['robot'].get('speed_y', 0.0)
        velocity = np.array([speed_x, speed_y])
        velocities.append(np.linalg.norm(velocity))
        
        # Accelerations
        if prev_velocity is not None and prev_time is not None and timestamp is not None:
            dv = velocity - prev_velocity
            dt = timestamp - prev_time
            if dt > 0:
                acceleration = np.linalg.norm(dv) / dt
                accelerations.append(acceleration)
                # Jerks
                if prev_acceleration is not None:
                    da = acceleration - prev_acceleration
                    jerk = da / dt
                    jerks.append(jerk)
                prev_acceleration = acceleration
        else:
            prev_acceleration = 0.0
        prev_velocity = velocity
        prev_time = timestamp
        
        # Clearing distance (distance to nearest human)
        min_distance = float('inf')
        for person in obs.get('people', []):
            agent_position = np.array([person['x'], person['y']])
            distance = np.linalg.norm(current_position - agent_position)
            if distance < min_distance:
                min_distance = distance
            if distance < min_distance_to_human:
                min_distance_to_human = distance
        if min_distance != float('inf'):
            clearing_distances.append(min_distance)
        
        # Time to collision (simplified)
        current_speed = velocities[-1]
        if current_speed > 0 and min_distance != float('inf'):
            ttc = min_distance / current_speed
            if ttc < min_time_to_collision:
                min_time_to_collision = ttc
    
    # Compute total time
    if timestamps:
        total_time = timestamps[-1] - timestamps[0]
    else:
        total_time = 0.0
    
    # Compute optimal path length (straight-line distance from start to goal)
    if sequence and 'robot' in sequence[0]:
        start_position = np.array([
            sequence[0]['robot']['x'],
            sequence[0]['robot']['y']
        ])
        goal_position = np.array([
            sequence[0]['robot']['goal_x'],
            sequence[0]['robot']['goal_y']
        ])
        optimal_path_length = np.linalg.norm(goal_position - start_position)
    else:
        # Handle the case where start or goal positions are missing
        optimal_path_length = 0.0
    
    # Success weighted by path length (SPL)
    if path_length > 0 and optimal_path_length > 0:
        spl = success * (optimal_path_length / max(path_length, optimal_path_length))
    else:
        spl = 0.0
    
    # Populate features
    features['filename'] = trajectory['filename']
    features['success'] = success
    features['collision_occurred'] = 1 if collision_occurred else 0
    features['wall_collisions'] = wall_collisions  # Placeholder
    features['agent_collisions'] = agent_collisions  # Placeholder
    features['human_collisions'] = human_collisions  # Placeholder
    features['timeout_occurred'] = timeout_occurred  # Placeholder
    features['failure_to_progress'] = failure_to_progress  # Placeholder
    features['stalled_time'] = stalled_time
    features['time_to_goal'] = total_time
    features['path_length'] = path_length
    features['success_weighted_path_length'] = spl
    
    # Velocity-based features
    features['v_min'] = min(velocities) if velocities else 0.0
    features['v_avg'] = np.mean(velocities) if velocities else 0.0
    features['v_max'] = max(velocities) if velocities else 0.0
    
    # Acceleration-based features
    features['a_min'] = min(accelerations) if accelerations else 0.0
    features['a_avg'] = np.mean(accelerations) if accelerations else 0.0
    features['a_max'] = max(accelerations) if accelerations else 0.0
    
    # Jerk-based features
    features['j_min'] = min(jerks) if jerks else 0.0
    features['j_avg'] = np.mean(jerks) if jerks else 0.0
    features['j_max'] = max(jerks) if jerks else 0.0
    
    # Clearing distance features
    features['cd_min'] = min(clearing_distances) if clearing_distances else 0.0
    features['cd_avg'] = np.mean(clearing_distances) if clearing_distances else 0.0
    
    # Space compliance (SC)
    features['space_compliance'] = acceptable_social_nav / 100.0 if acceptable_social_nav is not None else 0.0
    
    # Minimum distance to human
    features['min_distance_to_human'] = min_distance_to_human if min_distance_to_human != float('inf') else 0.0
    
    # Minimum time to collision
    features['min_time_to_collision'] = min_time_to_collision if min_time_to_collision != float('inf') else 0.0
    
    # Aggregated time (AT)
    features['aggregated_time'] = total_time  # Could be adjusted if needed
    
    # Quality score (label)
    if acceptable_social_nav is not None:
        quality_score = acceptable_social_nav / 100.0  # Normalise to [0, 1]
    else:
        quality_score = 0.0  # Default to 0 if not available
    features['quality_score'] = quality_score
    
    return features

def create_dataset(trajectories, labels):
    """
    Creates a dataset from the trajectories and labels.
    """
    data = []
    for traj in trajectories:
        filename = traj['filename']
        # Get the manual label
        manual_label = labels.get(filename, {})
        if 'acceptable_social_nav' not in manual_label:
            continue  # Skip trajectories without a quality score
        # Attach manual labels to trajectory
        traj['manual_label'] = manual_label
        features = extract_features(traj)
        data.append(features)
    df = pd.DataFrame(data)
    return df

def visualise_data(dataset):
    """
    Generates visualizations from the dataset.
    """
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    corr = dataset.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()
    
    # Distribution of Quality Scores
    plt.figure(figsize=(8, 6))
    sns.histplot(dataset['quality_score'], bins=20, kde=True)
    plt.title('Distribution of Quality Scores')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('quality_score_distribution.png')
    plt.show()
    
    # Scatter Plot of Path Length vs. Quality Score
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='path_length', y='quality_score', data=dataset)
    plt.title('Path Length vs. Quality Score')
    plt.xlabel('Path Length')
    plt.ylabel('Quality Score')
    plt.tight_layout()
    plt.savefig('path_length_vs_quality_score.png')
    plt.show()
    
    # Boxplot of Velocity Features
    velocity_features = ['v_min', 'v_avg', 'v_max']
    plt.figure(figsize=(10, 6))
    dataset_melted = dataset.melt(value_vars=velocity_features)
    sns.boxplot(x='variable', y='value', data=dataset_melted)
    plt.title('Velocity Features Distribution')
    plt.xlabel('Velocity Feature')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('velocity_features_boxplot.png')
    plt.show()
    
    # Feature Importance (if model is available)
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        feature_names = dataset.drop(['filename', 'quality_score'], axis=1).columns
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.show()
    else:
        print("Model file not found. Skipping feature importance visualization.")
    
    # Pair Plot of Selected Features
    selected_features = ['quality_score', 'v_avg', 'a_avg', 'cd_min', 'path_length']
    sns.pairplot(dataset[selected_features], diag_kind='kde')
    plt.suptitle('Pair Plot of Selected Features', y=1.02)
    plt.tight_layout()
    plt.savefig('pair_plot_selected_features.png')
    plt.show()
    
    # Save dataset to CSV
    dataset.to_csv('trajectory_dataset.csv', index=False)
    print("Dataset saved to 'trajectory_dataset.csv'.")

def main():
    # Load data
    print("Loading data...")
    trajectories = load_trajectory_data(DATA_DIR)
    labels = load_labels(LABELS_FILE)
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(trajectories, labels)
    
    print("Dataset Shape:", dataset.shape)
    print("Dataset Columns:", dataset.columns)
    
    # Handle missing values (if any)
    dataset = dataset.fillna(0)
    
    # Visualise data
    print("Visualizing data...")
    visualise_data(dataset)
    print("Visualization completed.")

if __name__ == '__main__':
    main()
