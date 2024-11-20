from flask import Flask, render_template, url_for, redirect, request, send_file
from trajectory_analyzer import load_trajectory_data
import json
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

DATA_DIR = './trajectory_dataset/'
LABELS_FILE = 'labels.json'

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
    return labels

def save_labels(labels):
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)

@app.template_filter('format_float')
def format_float(value):
    return f"{value:.2f}"

@app.route('/', methods=['GET', 'POST'])
def index():
    trajectories = load_trajectory_data(DATA_DIR)
    labels = load_labels()
    # Merge labels into trajectories
    for traj in trajectories:
        filename = traj['filename']
        traj['manual_label'] = labels.get(filename, {})

    # Filtering and Sorting
    filter_value = request.args.get('filter_value', type=float)
    sort_order = request.args.get('sort_order', 'asc')

    # Apply filtering
    if filter_value is not None:
        trajectories = [
            traj for traj in trajectories
            if traj['manual_label'].get('acceptable_social_nav') is not None and
               traj['manual_label']['acceptable_social_nav'] >= filter_value
        ]

    # Apply sorting
    trajectories.sort(key=lambda x: x['manual_label'].get('acceptable_social_nav', 0.0), reverse=(sort_order == 'desc'))

    # Statistical Analysis
    acceptable_values = [
        traj['manual_label']['acceptable_social_nav']
        for traj in trajectories
        if 'acceptable_social_nav' in traj['manual_label'] and traj['manual_label']['acceptable_social_nav'] is not None
    ]

    if acceptable_values:
        avg_value = np.mean(acceptable_values)
        median_value = np.median(acceptable_values)
        std_dev = np.std(acceptable_values)
    else:
        avg_value = median_value = std_dev = None

    # Generate visualization
    chart = create_bar_chart(trajectories)

    return render_template('results.html',
                           trajectories=trajectories,
                           filter_value=filter_value,
                           sort_order=sort_order,
                           avg_value=avg_value,
                           median_value=median_value,
                           std_dev=std_dev,
                           chart=chart)

def create_bar_chart(trajectories):
    # Prepare data
    filenames = [traj['filename'] for traj in trajectories]
    values = [
        traj['manual_label'].get('acceptable_social_nav', 0)
        for traj in trajectories
    ]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(filenames, values, color='skyblue')
    plt.xlabel('Trajectory File')
    plt.ylabel('Acceptable Social Behaviour')
    plt.title('Acceptable Social Behaviour Across Trajectories')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save chart to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image to base64 string
    chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return chart_data

@app.route('/trajectory/<filename>', methods=['GET', 'POST'])
def trajectory_detail(filename):
    trajectories = load_trajectory_data(DATA_DIR)
    trajectory = next((t for t in trajectories if t['filename'] == filename), None)
    if trajectory:
        labels = load_labels()
        manual_label = labels.get(filename, {})
        if request.method == 'POST':
            # Handle manual selection
            goal_reached = request.form.get('goal_reached') == 'on'
            collision_occurred = request.form.get('collision_occurred') == 'on'
            acceptable_social_nav = request.form.get('acceptable_social_nav')
            if acceptable_social_nav is not None:
                acceptable_social_nav = float(acceptable_social_nav)
            overall_rating = request.form.get('rating')
            if overall_rating in ['Success', 'Failure']:
                labels[filename] = {
                    'goal_reached': goal_reached,
                    'collision_occurred': collision_occurred,
                    'acceptable_social_nav': acceptable_social_nav,
                    'rating': overall_rating
                }
                save_labels(labels)
                manual_label = labels[filename]
        return render_template('trajectory_detail.html', trajectory=trajectory, manual_label=manual_label)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
