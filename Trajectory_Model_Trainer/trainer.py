import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON data
with open('trajectory_data.json', 'r') as f:
    data = json.load(f)

# Initialize lists to collect data
timestamps = []
SNGNN_values = []
robot_features = []
people_features = []

# Extract data from JSON
for entry in data['data']:
    timestamps.append(entry['timestamp'])
    SNGNN_values.append(entry['SNGNN'])
    
    robot = entry['robot']
    robot_state = {
        'robot_x': robot['x'],
        'robot_y': robot['y'],
        'robot_angle': robot['angle'],
        'robot_speed_x': robot['speed_x'],
        'robot_speed_y': robot['speed_y'],
        'robot_speed_a': robot['speed_a'],
        'goal_x': robot['goal_x'],
        'goal_y': robot['goal_y'],
    }
    robot_features.append(robot_state)
    
    # For each person, calculate features relative to the robot
    for person in entry['people']:
        person_state = {
            'timestamp': entry['timestamp'],
            'person_id': person['id'],
            'person_x': person['x'],
            'person_y': person['y'],
            'person_angle': person['angle'],
            'person_speed': person['speed'],
            'robot_x': robot['x'],
            'robot_y': robot['y'],
            'robot_angle': robot['angle'],
        }
        # Calculate derived features
        person_state['person_robot_dx'] = person['x'] - robot['x']
        person_state['person_robot_dy'] = person['y'] - robot['y']
        person_state['person_robot_distance'] = np.sqrt((person_state['person_robot_dx'])**2 + (person_state['person_robot_dy'])**2)
        person_state['person_robot_angle_diff'] = person['angle'] - robot['angle']
        # Normalize angle difference to [-pi, pi]
        person_state['person_robot_angle_diff'] = (person_state['person_robot_angle_diff'] + np.pi) % (2 * np.pi) - np.pi
        # Append to people_features
        people_features.append(person_state)

# Convert lists to DataFrames
robot_df = pd.DataFrame(robot_features)
people_df = pd.DataFrame(people_features)
SNGNN_series = pd.Series(SNGNN_values, name='SNGNN')

# Since people_df has multiple entries per timestamp, we need to aggregate or select relevant features
# Aggregate person features by calculating min, max, and mean distances and angle differences to the robot at each timestamp
people_grouped = people_df.groupby('timestamp').agg({
    'person_robot_distance': ['min', 'max', 'mean'],
    'person_robot_angle_diff': ['min', 'max', 'mean'],
})
# Flatten MultiIndex columns
people_grouped.columns = ['_'.join(col).strip() for col in people_grouped.columns.values]

# Merge robot_df and people_grouped on timestamps
robot_df['timestamp'] = timestamps
full_df = pd.merge(robot_df, people_grouped, left_index=True, right_index=True)
full_df['SNGNN'] = SNGNN_series

# Feature Engineering
# Calculate distance to goal
full_df['robot_goal_dx'] = full_df['goal_x'] - full_df['robot_x']
full_df['robot_goal_dy'] = full_df['goal_y'] - full_df['robot_y']
full_df['robot_goal_distance'] = np.sqrt(full_df['robot_goal_dx']**2 + full_df['robot_goal_dy']**2)

# Calculate heading error (difference between robot angle and angle to goal)
full_df['robot_goal_angle'] = np.arctan2(full_df['robot_goal_dy'], full_df['robot_goal_dx'])
full_df['heading_error'] = full_df['robot_goal_angle'] - full_df['robot_angle']

# Normalize angles to [-pi, pi]
full_df['heading_error'] = (full_df['heading_error'] + np.pi) % (2 * np.pi) - np.pi

# Calculate robot speed magnitude
full_df['robot_speed_magnitude'] = np.sqrt(full_df['robot_speed_x']**2 + full_df['robot_speed_y']**2)

# Handle missing values if any
full_df = full_df.dropna()

# Select features and target
features = [
    'robot_x', 'robot_y', 'robot_angle',
    'robot_speed_x', 'robot_speed_y', 'robot_speed_a',
    'robot_speed_magnitude',
    'robot_goal_distance', 'heading_error',
    'person_robot_distance_min', 'person_robot_distance_max', 'person_robot_distance_mean',
    'person_robot_angle_diff_min', 'person_robot_angle_diff_max', 'person_robot_angle_diff_mean',
]
X = full_df[features]
y = full_df['SNGNN']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# 1. Correlation Analysis
corr_matrix = full_df[features + ['SNGNN']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Identify features with high correlation to SNGNN
correlations = corr_matrix['SNGNN'].drop('SNGNN').sort_values(ascending=False)
print("Correlation with SNGNN:")
print(correlations)

# 2. Mutual Information
mi_scores = mutual_info_regression(X_scaled_df, y)
mi_scores = pd.Series(mi_scores, index=features)
mi_scores = mi_scores.sort_values(ascending=False)
print("\nMutual Information Scores:")
print(mi_scores)

# 3. Feature Importance from Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled_df, y)
importances = rf.feature_importances_
rf_importances = pd.Series(importances, index=features)
rf_importances = rf_importances.sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(rf_importances)

# Plotting Feature Importances
plt.figure(figsize=(12, 6))
rf_importances.plot(kind='bar')
plt.title('Feature Importances from Random Forest')
plt.ylabel('Importance')
plt.show()

# 4. Select Features Based on Metrics
# For example, select top features based on mutual information and feature importance
selected_features = mi_scores.index[:8].tolist()  # Select top 8 features
print("\nSelected Features for Modeling:")
print(selected_features)

# Prepare final dataset with selected features
X_selected = X_scaled_df[selected_features]

# Advanced Models Implementation

# Define a function to evaluate models using cross-validation
def evaluate_model(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()

# Use TimeSeriesSplit for time-series data
tscv = TimeSeriesSplit(n_splits=5)

# Dictionary to store models and their parameters for GridSearchCV
models = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear'],
        }
    },
    'NeuralNetwork': {
        'model': MLPRegressor(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01],
        }
    },
}

# Import XGBoost if available
try:
    from xgboost import XGBRegressor
    models['XGBoost'] = {
        'model': XGBRegressor(random_state=42, objective='reg:squarederror'),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
    }
except ImportError:
    print("XGBoost is not installed. Skipping XGBoost model.")

# Import LightGBM if available
try:
    from lightgbm import LGBMRegressor
    models['LightGBM'] = {
        'model': LGBMRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
        }
    }
except ImportError:
    print("LightGBM is not installed. Skipping LightGBM model.")

# Evaluate models
best_estimators = {}
for name, model_info in models.items():
    print(f"\nTraining {name} model...")
    grid_search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_selected, y)
    best_model = grid_search.best_estimator_
    rmse = np.sqrt(-grid_search.best_score_)
    print(f"Best RMSE for {name}: {rmse}")
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    best_estimators[name] = best_model

# Compare models
model_performance = {}
for name, estimator in best_estimators.items():
    rmse = evaluate_model(estimator, X_selected, y, tscv)
    model_performance[name] = rmse
    print(f"\n{name} RMSE: {rmse}")

# Plot model performance
plt.figure(figsize=(10, 5))
plt.bar(model_performance.keys(), model_performance.values())
plt.title('Model RMSE Comparison')
plt.ylabel('RMSE')
plt.show()

# Optional: Analyze feature importances for the best model (if applicable)
best_model_name = min(model_performance, key=model_performance.get)
print(f"\nBest Model: {best_model_name}")

if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
    best_model = best_estimators[best_model_name]
    feature_importances = pd.Series(best_model.feature_importances_, index=selected_features)
    feature_importances = feature_importances.sort_values(ascending=False)
    print(f"\nFeature Importances from {best_model_name}:")
    print(feature_importances)
    
    # Plotting Feature Importances
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title(f'Feature Importances from {best_model_name}')
    plt.ylabel('Importance')
    plt.show()

# Predictions and Residual Analysis
# Retrain the best model on the entire dataset
best_model = best_estimators[best_model_name]
best_model.fit(X_selected, y)

# Make predictions
y_pred = best_model.predict(X_selected)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(full_df['timestamp'], y, label='Actual SNGNN', marker='o')
plt.plot(full_df['timestamp'], y_pred, label='Predicted SNGNN', marker='x')
plt.xlabel('Timestamp')
plt.ylabel('SNGNN')
plt.title('Actual vs Predicted SNGNN Over Time')
plt.legend()
plt.show()

# Residual Plot
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red')
plt.xlabel('Predicted SNGNN')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Save the best model if needed
import joblib
joblib.dump(best_model, f'{best_model_name}_best_model.pkl')
