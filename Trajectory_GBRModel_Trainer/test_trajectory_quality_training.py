import unittest
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open

# Import the functions from trajectory_quality_training.py
from trajectory_quality_training import (
    load_trajectory_data,
    load_labels,
    extract_features,
    create_dataset,
    main
)

class TestTrajectoryQualityTraining(unittest.TestCase):

    def setUp(self):
        # Sample trajectory data for A000001.json
        self.sample_trajectory = {
            'filename': 'A000001.json',
            'data': {
                'sequence': [
                    {
                        'timestamp': 0.0,
                        'robot': {
                            'x': 0.0,
                            'y': 0.0,
                            'speed_x': 0.0,
                            'speed_y': 0.0,
                            'goal_x': 5.0,
                            'goal_y': 5.0
                        },
                        'people': [
                            {'x': 1.0, 'y': 1.0},
                            {'x': 2.0, 'y': 2.0}
                        ]
                    },
                    {
                        'timestamp': 1.0,
                        'robot': {
                            'x': 1.0,
                            'y': 1.0,
                            'speed_x': 1.0,
                            'speed_y': 1.0
                        },
                        'people': [
                            {'x': 1.5, 'y': 1.5},
                            {'x': 2.5, 'y': 2.5}
                        ]
                    },
                    # Additional observations can be added here
                ]
            }
        }

        # Sample labels data
        self.sample_labels = {
            'A000001.json': {
                'goal_reached': True,
                'collision_occurred': False,
                'acceptable_social_nav': 95.0,
                'rating': 'Success'
            },
            'A000002.json': {
                'goal_reached': False,
                'collision_occurred': True,
                'acceptable_social_nav': 20.0,
                'rating': 'Failure'
            }
            # Additional labels can be added here
        }

    def test_load_trajectory_data(self):
        # Mock os.listdir to return a predefined list of files
        with patch('os.listdir', return_value=['A000001.json']):
            # Mock open to read sample trajectory data
            with patch('builtins.open', mock_open(read_data=json.dumps(self.sample_trajectory['data']))):
                trajectories = load_trajectory_data('dummy_directory')
                self.assertEqual(len(trajectories), 1)
                self.assertEqual(trajectories[0]['filename'], 'A000001.json')
                self.assertIn('data', trajectories[0])

    def test_load_labels(self):
        # Mock open to read sample labels data
        with patch('builtins.open', mock_open(read_data=json.dumps(self.sample_labels))):
            labels = load_labels('dummy_labels.json')
            self.assertEqual(len(labels), 2)
            self.assertIn('A000001.json', labels)
            self.assertIn('rating', labels['A000001.json'])
            self.assertEqual(labels['A000001.json']['rating'], 'Success')

    def test_extract_features(self):
        # Add manual label to trajectory
        self.sample_trajectory['manual_label'] = self.sample_labels['A000001.json']
        features = extract_features(self.sample_trajectory)

        # Check that features are extracted correctly
        self.assertIn('success', features)
        self.assertEqual(features['success'], 1)  # 'Success' rating corresponds to success = 1
        self.assertIn('collision_occurred', features)
        self.assertEqual(features['collision_occurred'], 0)
        self.assertIn('space_compliance', features)
        self.assertAlmostEqual(features['space_compliance'], 0.95)
        self.assertIn('quality_score', features)
        self.assertAlmostEqual(features['quality_score'], 0.95)
        # Add more assertions for other features as needed

    def test_create_dataset(self):
        # Prepare sample trajectories and labels
        trajectories = [self.sample_trajectory]
        labels = self.sample_labels

        # Create dataset
        dataset = create_dataset(trajectories, labels)

        # Check that the dataset has the correct shape and columns
        self.assertEqual(len(dataset), 1)
        expected_columns = [
            'success', 'collision_occurred', 'wall_collisions', 'agent_collisions',
            'human_collisions', 'timeout_occurred', 'failure_to_progress',
            'stalled_time', 'time_to_goal', 'path_length',
            'success_weighted_path_length', 'v_min', 'v_avg', 'v_max', 'a_min',
            'a_avg', 'a_max', 'j_min', 'j_avg', 'j_max', 'cd_min', 'cd_avg',
            'space_compliance', 'min_distance_to_human', 'min_time_to_collision',
            'aggregated_time', 'quality_score'
        ]
        self.assertListEqual(dataset.columns.tolist(), expected_columns)

    def test_main_function(self):
        # Prepare multiple samples for the test
        sample_features_1 = extract_features({
            'filename': 'A000001.json',
            'data': self.sample_trajectory['data'],
            'manual_label': self.sample_labels['A000001.json']
        })
        # Create a second sample trajectory
        sample_trajectory_2 = {
            'filename': 'A000002.json',
            'data': {
                'sequence': [
                    {
                        'timestamp': 0.0,
                        'robot': {
                            'x': 0.0,
                            'y': 0.0,
                            'speed_x': 0.0,
                            'speed_y': 0.0,
                            'goal_x': 5.0,
                            'goal_y': 5.0
                        },
                        'people': []
                    },
                    {
                        'timestamp': 1.0,
                        'robot': {
                            'x': 0.5,
                            'y': 0.5,
                            'speed_x': 0.5,
                            'speed_y': 0.5
                        },
                        'people': []
                    },
                ]
            },
            'manual_label': self.sample_labels['A000002.json']
        }
        sample_features_2 = extract_features(sample_trajectory_2)

        # Mock functions and methods to test main without actual file operations
        with patch('trajectory_quality_training.load_trajectory_data') as mock_load_data, \
             patch('trajectory_quality_training.load_labels') as mock_load_labels, \
             patch('trajectory_quality_training.create_dataset') as mock_create_dataset, \
             patch('trajectory_quality_training.train_test_split') as mock_train_test_split, \
             patch('trajectory_quality_training.StandardScaler') as mock_scaler_class, \
             patch('trajectory_quality_training.GradientBoostingRegressor') as mock_gbr_class, \
             patch('trajectory_quality_training.cross_val_score') as mock_cross_val_score, \
             patch('trajectory_quality_training.joblib.dump') as mock_joblib_dump:

            # Mock the data returned by load_trajectory_data and load_labels
            mock_load_data.return_value = [self.sample_trajectory, sample_trajectory_2]
            mock_load_labels.return_value = self.sample_labels

            # Mock the dataset with two samples
            sample_dataset = pd.DataFrame([sample_features_1, sample_features_2])
            mock_create_dataset.return_value = sample_dataset

            # Mock train_test_split
            X = sample_dataset.drop('quality_score', axis=1)
            y = sample_dataset['quality_score']
            mock_train_test_split.return_value = (X, X, y, y)

            # Mock StandardScaler
            mock_scaler = mock_scaler_class.return_value
            mock_scaler.fit_transform.return_value = X.values
            mock_scaler.transform.return_value = X.values

            # Mock GradientBoostingRegressor
            mock_gbr = mock_gbr_class.return_value
            mock_gbr.predict.return_value = y.values
            # Mock feature_importances_
            mock_gbr.feature_importances_ = np.array([1 / len(X.columns)] * len(X.columns))

            # Mock cross_val_score
            mock_cross_val_score.return_value = np.array([-0.0001, -0.0002, -0.0001, -0.00015, -0.0001])

            # Run main
            main()

            # Assertions to ensure functions were called
            mock_load_data.assert_called_once()
            mock_load_labels.assert_called_once()
            mock_create_dataset.assert_called_once()
            mock_train_test_split.assert_called_once()
            mock_scaler_class.assert_called_once()
            mock_gbr_class.assert_called_once()
            mock_cross_val_score.assert_called_once()
            mock_joblib_dump.assert_called()

    def test_create_dataset_skips_incomplete_labels(self):
        # Create labels missing 'acceptable_social_nav' and 'rating'
        incomplete_labels = {
            'A000001.json': {
                'goal_reached': True,
                'collision_occurred': False
                # 'acceptable_social_nav' and 'rating' are missing
            }
        }
        trajectories = [self.sample_trajectory]
        dataset = create_dataset(trajectories, incomplete_labels)
        # Dataset should be empty as the trajectory lacks necessary labels
        self.assertEqual(len(dataset), 0)

    def test_extract_features_handles_missing_data(self):
        # Remove 'people' from observations
        self.sample_trajectory['data']['sequence'][0].pop('people', None)
        self.sample_trajectory['data']['sequence'][1].pop('people', None)
        self.sample_trajectory['manual_label'] = self.sample_labels['A000001.json']
        features = extract_features(self.sample_trajectory)
        # Features related to 'people' should be handled gracefully
        self.assertIn('cd_min', features)
        self.assertEqual(features['cd_min'], 0.0)
        self.assertIn('cd_avg', features)
        self.assertEqual(features['cd_avg'], 0.0)
        # 'space_compliance' remains calculated based on manual labels
        self.assertIn('space_compliance', features)
        self.assertAlmostEqual(features['space_compliance'], 0.95)

if __name__ == '__main__':
    unittest.main()
