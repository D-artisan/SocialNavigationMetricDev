import os
import json
import numpy as np
import pandas as pd

# Directories and files
DATA_DIR = './testing_dataset/'
OUTPUT_FILE = './new_features_dataset.csv'

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

    # Flags (set defaults or extract from data if available)
    goal_reached = data.get('goal_reached', False)
    collision_occurred = data.get('collision_occurred', False)
    acceptable_social_nav = None  # Not available in new data
    rating = None  # Not available in new data

    # Estimate 'success' based on available data
    # Assuming that if the robot reached the goal without collision, it's a success
    success = 1 if goal_reached and not collision_occurred else 0

    # Since 'acceptable_social_nav' is not available, we'll estimate 'space_compliance'
    # Define a threshold for comfortable distance (e.g., 0.5 meters)
    comfortable_distance = 0.5

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
            sequence[0]['robot'].get('goal_x', 0.0),
            sequence[0]['robot'].get('goal_y', 0.0)
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

    # Estimate space compliance based on minimum distance to humans
    if min_distance_to_human != float('inf'):
        if min_distance_to_human >= comfortable_distance:
            space_compliance = 1.0  # Full compliance
        else:
            # Decrease compliance proportionally to how much the robot encroaches on personal space
            space_compliance = min_distance_to_human / comfortable_distance
    else:
        # No humans encountered; assume full compliance
        space_compliance = 1.0

    # Populate features
    features['success'] = success
    features['collision_occurred'] = 1 if collision_occurred else 0
    features['wall_collisions'] = wall_collisions  # Placeholder or extract if available
    features['agent_collisions'] = agent_collisions  # Placeholder or extract if available
    features['human_collisions'] = human_collisions  # Placeholder or extract if available
    features['timeout_occurred'] = 1 if not goal_reached and not collision_occurred else 0
    features['failure_to_progress'] = failure_to_progress  # Placeholder or determine based on data
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
    features['cd_min'] = min(clearing_distances) if clearing_distances else comfortable_distance
    features['cd_avg'] = np.mean(clearing_distances) if clearing_distances else comfortable_distance

    # Use the estimated 'space_compliance'
    features['space_compliance'] = space_compliance

    # Minimum distance to human
    features['min_distance_to_human'] = min_distance_to_human if min_distance_to_human != float('inf') else comfortable_distance

    # Minimum time to collision
    features['min_time_to_collision'] = min_time_to_collision if min_time_to_collision != float('inf') else total_time

    # Aggregated time (AT)
    features['aggregated_time'] = total_time

    # Exclude 'quality_score' as it is unknown
    # features['quality_score'] = None  # Do not include in features

    return features

def create_dataset(trajectories):
    """
    Creates a dataset from the trajectories.
    """
    data = []
    for traj in trajectories:
        features = extract_features(traj)
        data.append(features)
    df = pd.DataFrame(data)
    return df

def main():
    # Load trajectory data
    print("Loading trajectory data...")
    trajectories = load_trajectory_data(DATA_DIR)

    # Create dataset
    print("Extracting features...")
    dataset = create_dataset(trajectories)

    print("Dataset Shape:", dataset.shape)
    print("Dataset Columns:", dataset.columns)

    # Handle missing values (if any)
    dataset = dataset.fillna(0)

    # Save features to CSV
    print(f"Saving extracted features to {OUTPUT_FILE}...")
    dataset.to_csv(OUTPUT_FILE, index=False)
    print("Feature extraction completed.")

if __name__ == '__main__':
    main()
