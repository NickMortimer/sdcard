

import argparse
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
from pathlib import Path
from sdcard.config import Config

parser = argparse.ArgumentParser(description="Video Waypoint Browser Flask App")
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args, unknown = parser.parse_known_args()
DEBUG_MODE = args.debug

app = Flask(__name__)

# Load config and deployments CSV
config = Config()
deployments_path = config.get_path('deployments')
if deployments_path.exists():
    deployments = pd.read_csv(config.get_path('videolist'))
else:
    deployments = pd.DataFrame([])

# Load waypoints CSV (shared for all videos, or customize per video)
waypoints_csv = config.get_path('gps_folder') / 'waypoints.csv'
if waypoints_csv.exists():
    waypoints = pd.read_csv(waypoints_csv).to_dict(orient='records')
else:
    waypoints = []

@app.route("/")
def index():
    if deployments.empty:
        return "No deployments found."
    # Pass all video data and waypoints to the template
    return render_template(
        "video_waypoint_browser.html",
        deployments=deployments.to_dict(orient='records'),
        waypoints=waypoints
    )


@app.route("/video/<path:filename>")
def video(filename):
    video_path =Path("/" + filename) if not filename.startswith("/") else Path(filename)
    if DEBUG_MODE:
        print(f"[DEBUG] Serving video: {video_path} (exists: {video_path.exists()})")
    if not video_path.exists():
        return f"[ERROR] Video file not found: {video_path}", 404
    return send_from_directory(video_path.parent, video_path.name)


@app.route("/select", methods=["POST"])
def select():
    video_idx = int(request.form.get('video_idx', 0))
    selected_waypoint = request.form.get("waypoint")
    # Save or process the selection as needed (e.g., write to a file or DB)
    return redirect(url_for('index'))

if __name__ == "__main__":
    print(f"[DEBUG] Flask app running in debug={DEBUG_MODE}")
    app.run(debug=DEBUG_MODE)
