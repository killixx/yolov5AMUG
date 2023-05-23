from flask import Flask, render_template, request
from aumg import video_process
import os

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Initialize the video processing class
video_processor = video_process()

# Define a function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/AUMG')
def AUMG():
    return render_template('AUMG.html')

@app.route('/generateHighlights', methods=['POST'])
def upload():
    # Check if a file is selected in the request
    if 'file' not in request.files:
        return 'No file selected.'

    file = request.files['file']

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
       
        # Get the saved file path
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # Call the run function of the video processing class
        video_processor.run(video_path)

        return 'Video processing started.'

    return 'Invalid file format.'

if __name__ == '__main__':
    app.run(debug=True)