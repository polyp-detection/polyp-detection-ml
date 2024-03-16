from flask import Flask, render_template, request, jsonify
import os
from final1 import process_image, process_polyp_detection

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                os.makedirs('uploads', exist_ok=True)
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                result = process_image(file_path)
                return jsonify(result)
            else:
                return jsonify({'error': 'No file uploaded'})
        except Exception as e:
            return jsonify({'error': 'An error occurred while processing the file'})

@app.route('/detect_polyp', methods=['POST'])
def detect_polyp():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                os.makedirs('uploads', exist_ok=True)
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)
                result = process_polyp_detection(file_path)
                return jsonify(result)
            else:
                return jsonify({'error': 'No file uploaded'})
        except Exception as e:
            return jsonify({'error': 'An error occurred while processing the polyp detection'})

if __name__ == '__main__':
    app.run(debug=True)
