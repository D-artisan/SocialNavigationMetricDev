import os
import json

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
