from flask import Flask, render_template, url_for, redirect, request, send_from_directory
from trajectory_analyzer import load_trajectory_data
import json
import os
import matplotlib.pyplot as plt
import io
import base64
from jinja2 import Undefined

app = Flask(__name__)

DATA_DIR = './trajectory_dataset/'
LABELS_FILE = 'labels.json'
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.webm']  

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
        return labels
    else:
        return {}

def save_labels(labels):
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)

@app.template_filter('format_float')
def format_float(value):
    if value is not None and not isinstance(value, Undefined):
        return f"{value:.2f}"
    else:
        return "N/A"

@app.route('/')
def index():
    trajectories = load_trajectory_data(DATA_DIR)
    labels = load_labels()
    # Merge labels into trajectories
    for traj in trajectories:
        filename = traj['filename']
        traj['manual_label'] = labels.get(filename, {})
        # Ensure manual_label is a dictionary
        if traj['manual_label'] is None:
            traj['manual_label'] = {}
        # Find corresponding video file
        base_name = os.path.splitext(filename)[0]
        video_file = None
        for ext in VIDEO_EXTENSIONS:
            potential_video = base_name + ext
            if os.path.exists(os.path.join(DATA_DIR, potential_video)):
                video_file = potential_video
                break
        traj['video_file'] = video_file
    return render_template('results.html', trajectories=trajectories)

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
            else:
                acceptable_social_nav = None
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
        # Ensure manual_label is a dictionary
        if manual_label is None:
            manual_label = {}
        # Find corresponding video file
        base_name = os.path.splitext(filename)[0]
        video_file = None
        for ext in VIDEO_EXTENSIONS:
            potential_video = base_name + ext
            if os.path.exists(os.path.join(DATA_DIR, potential_video)):
                video_file = potential_video
                break
        trajectory['video_file'] = video_file

        # Generate bar chart (e.g., Discomfort Over Time)
        timestamps = [obs['timestamp'] for obs in trajectory['data']['sequence']]
        discomfort_values = [obs.get('SNGNN', 0.0) for obs in trajectory['data']['sequence']]

        plt.figure(figsize=(10, 4))
        plt.bar(timestamps, discomfort_values)
        plt.xlabel('Timestamp')
        plt.ylabel('Discomfort (SNGNN)')
        plt.title('Discomfort Over Time')
        plt.tight_layout()

        # Save plot to a string buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close()

        return render_template('trajectory_detail.html', trajectory=trajectory, manual_label=manual_label, plot_url=plot_url)
    else:
        return redirect(url_for('index'))

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(DATA_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
