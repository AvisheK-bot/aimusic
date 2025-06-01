import os
import subprocess
import sys
from flask import Flask

application = Flask(__name__)

@application.route('/')
def index():
    # Start Streamlit in a separate process
    process = subprocess.Popen([
        sys.executable,
        "-m", "streamlit", "run",
        "app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
    return "Streamlit app is running on port 8501"

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000) 