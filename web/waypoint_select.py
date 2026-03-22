from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Configurable paths (replace with your actual paths or config logic)
VIDEO_PATH = Path("/path/to/your/video.mp4")
WAYPOINTS_CSV = Path("/path/to/your/waypoints.csv")

@app.route("/")
def index():
    # Load waypoints from CSV
    if WAYPOINTS_CSV.exists():
        waypoints = pd.read_csv(WAYPOINTS_CSV).to_dict(orient='records')
    else:
        waypoints = []
    return render_template("waypoint_select.html", video_file=VIDEO_PATH.name, waypoints=waypoints)

@app.route("/video/<filename>")
def video(filename):
    return send_from_directory(VIDEO_PATH.parent, filename)

@app.route("/select", methods=["POST"])
def select():
    selected = request.form.get("waypoint")
    # Here you could save the selection, process it, etc.
    return f"You selected waypoint: {selected}"

if __name__ == "__main__":
    app.run(debug=True)
