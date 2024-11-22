import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import pandas as pd
import json
import joblib

# Import the functions from trajectory_quality_training.py
# Adjust the import statement based on your project structure
from trajectory_quality_training import (
    load_trajectory_data,
    load_labels,
    extract_features,
    create_dataset,
    predict_trajectory_quality
)

class TestTrajectoryQualityTraining(unittest.TestCase):

    @patch('os.listdir')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_trajectory_data(self, mock_json_load, mock_open_func, mock_listdir):
        # Mock the list of files returned by os.listdir
        mock_listdir.return_value = ['trajectory1.json', 'trajectory2.json']
        
        # Mock the data returned by json.load
        mock_json_load.side_effect = [
            {'sequence': 'trajectory_data_1'},
            {'sequence': 'trajectory_data_2'}
        ]
        
        # Call the function
        trajectories = load_trajectory_data('./trajectory_dataset/')
        
        # Assertions
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0]['filename'], 'trajectory1.json')
        self.assertEqual(trajectories[0]['data'], {'sequence': 'trajectory_data_1'})
        self.assertEqual(trajectories[1]['filename'], 'trajectory2.json')
        self.assertEqual(trajectories[1]['data'], {'sequence': 'trajectory_data_2'})

    @patch('builtins.open')
    @patch('json.load')
    def test_load_labels(self, mock_json_load, mock_open_func):
        # Mock the data returned by json.load
        mock_json_load.return_value = {
            'trajectory1.json': {'acceptable_social_nav': 80},
            'trajectory2.json': {'acceptable_social_nav': 90}
        }
        
        # Call the function
        labels = load_labels('labels.json')
        
        # Assertions
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels['trajectory1.json']['acceptable_social_nav'], 80)
        self.assertEqual(labels['trajectory2.json']['acceptable_social_nav'], 90)

    def test_extract_features(self):
        # Create a mock trajectory
        trajectory = {
            'filename': 'trajectory1.json',
            'data': {
                'sequence': [
                    {
                        'timestamp': 0.0,
                        'robot': {
                            'x': 0.0,
                            'y': 0.0,
                            'goal_x': 10.0,
                            'goal_y': 0.0,
                            'speed_x': 0.0,
                            'speed_y': 0.0
                        },
                        'people': [
                            {'x': 5.0, 'y': 0.0}
                        ]
                    },
                    {
                        'timestamp': 1.0,
                        'robot': {
                            'x': 1.0,
                            'y': 0.0,
                            'goal_x': 10.0,
                            'goal_y': 0.0,
                            'speed_x': 1.0,
                            'speed_y': 0.0
                        },
                        'people': [
                            {'x': 5.0, 'y': 0.0}
                        ]
                    },
                    {
                        'timestamp': 2.0,
                        'robot': {
                            'x': 3.0,
                            'y': 0.0,
                            'goal_x': 10.0,
                            'goal_y': 0.0,
                            'speed_x': 2.0,
                            'speed_y': 0.0
                        },
                        'people': [
                            {'x': 5.0, 'y': 0.0}
                        ]
                    }
                ]
            },
            'manual_label': {
                'goal_reached': True,
                'collision_occurred': False,
                'acceptable_social_nav': 85
            }
        }
        
        # Call the function
        features = extract_features(trajectory)
        
        # Assertions
        self.assertEqual(features['success'], 1)
        self.assertEqual(features['collision_occurred'], 0)
        self.assertAlmostEqual(features['path_length'], 3.0)
        self.assertAlmostEqual(features['time_to_goal'], 2.0)
        self.assertEqual(features['quality_score'], 0.85)
        # Updated expectations to match the function's behavior
        self.assertEqual(features['v_min'], 0.0)           # Expected minimum velocity is 0.0
        self.assertEqual(features['v_max'], 2.0)
        self.assertAlmostEqual(features['v_avg'], 1.0)     # Average velocity is (0.0 + 1.0 + 2.0) / 3 = 1.0
        self.assertTrue('a_min' in features)
        self.assertTrue('a_max' in features)
        self.assertTrue('a_avg' in features)
        self.assertTrue('j_min' in features)
        self.assertTrue('j_max' in features)
        self.assertTrue('j_avg' in features)
        self.assertTrue('cd_min' in features)
        self.assertTrue('cd_avg' in features)
        self.assertAlmostEqual(features['min_distance_to_human'], 2.0)
        self.assertTrue('min_time_to_collision' in features)

    def test_create_dataset(self):
        # Mock trajectories and labels
        trajectories = [
            {
                'filename': 'trajectory1.json',
                'data': {
                    'sequence': [
                        {
                            'timestamp': 0.0,
                            'robot': {
                                'x': 0.0,
                                'y': 0.0,
                                'goal_x': 10.0,
                                'goal_y': 0.0,
                                'speed_x': 0.0,
                                'speed_y': 0.0
                            },
                            'people': []
                        }
                    ]
                }
            },
            {
                'filename': 'trajectory2.json',
                'data': {
                    'sequence': []
                }
            }
        ]
        labels = {
            'trajectory1.json': {
                'acceptable_social_nav': 80,
                'goal_reached': True,
                'collision_occurred': False
            },
            'trajectory2.json': {
                'acceptable_social_nav': 90,
                'goal_reached': False,
                'collision_occurred': True
            }
        }
        
        # Call the function
        dataset = create_dataset(trajectories, labels)
        
        # Assertions
        self.assertEqual(len(dataset), 2)
        self.assertIn('quality_score', dataset.columns)
        self.assertEqual(dataset.iloc[0]['quality_score'], 0.8)
        self.assertEqual(dataset.iloc[1]['quality_score'], 0.9)

    @patch('joblib.load')
    def test_predict_trajectory_quality(self, mock_joblib_load):
        # Mock the scaler and model
        mock_scaler = MagicMock()
        mock_model = MagicMock()
        
        # Mock the scaler's transform method
        mock_scaler.transform.return_value = np.array([[0.0] * 25])  # Adjust the number based on feature count
        
        # Mock the model's predict method
        mock_model.predict.return_value = [0.75]
        
        # Set the side effect of joblib.load
        def joblib_load_side_effect(filename):
            if 'scaler.pkl' in filename:
                return mock_scaler
            elif 'trajectory_quality_model.pkl' in filename:
                return mock_model
            else:
                return None
        
        mock_joblib_load.side_effect = joblib_load_side_effect
        
        # Create a mock trajectory
        trajectory = {
            'filename': 'trajectory1.json',
            'data': {
                'sequence': [
                    {
                        'timestamp': 0.0,
                        'robot': {
                            'x': 0.0,
                            'y': 0.0,
                            'goal_x': 10.0,
                            'goal_y': 0.0,
                            'speed_x': 0.0,
                            'speed_y': 0.0
                        },
                        'people': []
                    }
                ]
            },
            'manual_label': {
                'goal_reached': True,
                'collision_occurred': False,
                'acceptable_social_nav': 85
            }
        }
        
        # Call the function
        quality_score = predict_trajectory_quality(trajectory)
        
        # Assertions
        self.assertEqual(quality_score, 0.75)
        mock_scaler.transform.assert_called_once()
        mock_model.predict.assert_called_once()

    def test_extract_features_missing_data(self):
        # Create a trajectory with missing keys
        trajectory = {
            'filename': 'trajectory_missing.json',
            'data': {
                'sequence': []
            },
            'manual_label': {
                'goal_reached': False,
                'collision_occurred': False,
                'acceptable_social_nav': None
            }
        }
        
        # Call the function
        features = extract_features(trajectory)
        
        # Assertions
        self.assertEqual(features['success'], 0)
        self.assertEqual(features['collision_occurred'], 0)
        self.assertEqual(features['path_length'], 0.0)
        self.assertEqual(features['time_to_goal'], 0.0)
        self.assertEqual(features['quality_score'], 0.0)
        self.assertEqual(features['v_min'], 0.0)
        self.assertEqual(features['v_max'], 0.0)
        self.assertEqual(features['v_avg'], 0.0)
        self.assertEqual(features['cd_min'], 0.0)
        self.assertEqual(features['cd_avg'], 0.0)
        self.assertEqual(features['min_distance_to_human'], 0.0)

    @patch('os.listdir')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_trajectory_data_no_files(self, mock_json_load, mock_open_func, mock_listdir):
        # Mock the list of files returned by os.listdir to be empty
        mock_listdir.return_value = []
        
        # Call the function
        trajectories = load_trajectory_data('./trajectory_dataset/')
        
        # Assertions
        self.assertEqual(len(trajectories), 0)

    def test_create_dataset_no_labels(self):
        # Mock trajectories
        trajectories = [
            {
                'filename': 'trajectory1.json',
                'data': {
                    'sequence': []
                }
            }
        ]
        labels = {}  # Empty labels
        
        # Call the function
        dataset = create_dataset(trajectories, labels)
        
        # Assertions
        self.assertEqual(len(dataset), 0)

if __name__ == '__main__':
    unittest.main()
