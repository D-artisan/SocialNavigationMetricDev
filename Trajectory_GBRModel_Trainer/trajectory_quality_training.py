import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Directories and files
DATA_DIR = './trajectory_dataset/'
LABELS_FILE = 'labels.json'

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
    # Use 'rating' to determine success
    success = 1 if manual_label.get('rating', '') == 'Success' else 0
    collision_occurred = manual_label.get('collision_occurred', False)
    acceptable_social_nav = manual_label.get('acceptable_social_nav', None)
    goal_reached = manual_label.get('goal_reached', False)

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

        # Clearing distance (distance to nearest obstacle/human)
        min_distance = float('inf')
        for person in obs.get('people', []):
            agent_position = np.array([person['x'], person['y']])
            distance = np.linalg.norm(current_position - agent_position)
            if distance < min_distance:
                min_distance = distance
            # Update min_distance_to_human
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
    features['success'] = success
    features['collision_occurred'] = 1 if collision_occurred else 0
    features['wall_collisions'] = 0  # Placeholder or extract if available
    features['agent_collisions'] = 0  # Placeholder or extract if available
    features['human_collisions'] = 0  # Placeholder or extract if available
    features['timeout_occurred'] = 1 if not goal_reached and not collision_occurred else 0
    features['failure_to_progress'] = 0  # Placeholder or determine based on data
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
        if 'acceptable_social_nav' not in manual_label or 'rating' not in manual_label:
            continue  # Skip trajectories without necessary labels
        # Attach manual labels to trajectory
        traj['manual_label'] = manual_label
        features = extract_features(traj)
        data.append(features)
    df = pd.DataFrame(data)
    return df

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

    # Features and target variable
    X = dataset.drop('quality_score', axis=1)
    y = dataset['quality_score']

    # List of feature names
    feature_names = X.columns.tolist()

    # Handle missing values (if any)
    X = X.fillna(0)

    # Split the dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data preprocessing
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    print("Training the model...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 1)  # Ensure predictions are between 0 and 1
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    print("Cross-Validation MSE:", cv_mse)

    # Feature importance
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    print("Feature Importance:")
    print(feature_importance)

    # Save the trained model and scaler
    print("Saving the model and scaler...")
    os.makedirs('./training_output/', exist_ok=True)
    joblib.dump(model, './training_output/trajectory_quality_model.pkl')
    joblib.dump(scaler, './training_output/scaler.pkl')
    print("Model and scaler saved.")

if __name__ == '__main__':
    main()
