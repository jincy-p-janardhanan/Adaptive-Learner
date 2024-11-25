# flask web app for data entry to prepare the dataset. 
# The app is deployed at https://adaptive-learner.onrender.com/

from flask import Flask, request, render_template, jsonify, send_file
import json
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# To access dataset.json file from AWS storage
BUCKET_NAME = "adaptive-learner-bucket"
JSON_FILE_KEY = "dataset.json"
s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION'), 
                         aws_access_key_id=os.getenv('ACCESS_KEY_ID'), 
                         aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'))

# helper functions to load data from AWS S3 bucket
def load_data_from_s3():
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=JSON_FILE_KEY)
        data = json.load(response['Body'])
        return data
    except s3_client.exceptions.NoSuchKey:
        return []

# helper functions to save data to AWS S3 bucket    
def save_data_to_s3(data):
    json_content = json.dumps(data, indent=4)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=JSON_FILE_KEY, Body=json_content, ContentType='application/json')

# load data from the json file
data = load_data_from_s3()


# API CALLS

# load the web form
@app.route('/')
def index():
    return render_template('data-entry.html', data=data)

# Add new entry to the json file
@app.route('/add_entry', methods=['POST'])
def add_entry():
    
    next_id = max((entry['id'] for entry in data), default=0) + 1
    
    difficulty = request.form.get("difficulty", type=int)
    title = request.form.get("title")
    type_ = request.form.get("type")
    prologue = request.form.get("prologue")
    content = request.form.get("content")
    new_words = json.loads(request.form.get("new_words"))
    exercises = json.loads(request.form.get("exercises"))
    
    new_entry = {
        "id": next_id,
        "difficulty": difficulty,
        "type": type_,
        "title": title,
        "prologue": prologue,
        "content": content,
        "new_words": new_words,
        "exercises": exercises
    }
    data.append(new_entry)
    save_data_to_s3(data)

    return jsonify({"success": True, "message": "Entry added successfully!"})

# search and load chapter for editing
@app.route('/get_chapter', methods=['GET'])
def get_chapter():
    search_query = request.args.get('query', '').strip()
    if not search_query:
        return jsonify({"success": False, "message": "Query cannot be empty"}), 400
    
    for chapter in data:
        if str(chapter['id']) == search_query or chapter['title'].lower() == search_query.lower():
            return jsonify({"success": True, "chapter": chapter}), 200

    return jsonify({"success": False, "message": "Chapter not found"}), 404

# update chapter
@app.route('/update_entry', methods=['POST'])
def update_entry():
    chapter_id = request.form.get("id", type=int)
    if not chapter_id:
        return jsonify({"success": False, "message": "Chapter ID is required"}), 400

    difficulty = request.form.get("difficulty", type=int)
    title = request.form.get("title")
    type_ = request.form.get("type")
    prologue = request.form.get("prologue")
    content = request.form.get("content")
    new_words = json.loads(request.form.get("new_words"))
    exercises = json.loads(request.form.get("exercises"))

    for chapter in data:
        if chapter['id'] == chapter_id:
            chapter.update({
                "difficulty": difficulty,
                "type": type_,
                "title": title,
                "prologue": prologue,
                "content": content,
                "new_words": new_words,
                "exercises": exercises
            })
            save_data_to_s3(data)

            return jsonify({"success": True, "message": "Chapter updated successfully!"}), 200

    return jsonify({"success": False, "message": "Chapter not found!"}), 404

# download the dataset
@app.route('/dataset', methods=['GET'])
def download_dataset():
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=JSON_FILE_KEY)
        json_data = response['Body'].read()
        
        return send_file(BytesIO(json_data), as_attachment=True, download_name=JSON_FILE_KEY, mimetype='application/json')
    
    except s3_client.exceptions.NoSuchKey:
        return jsonify({"message": "Dataset not found!"}), 404

if __name__ == "__main__":
    app.run(debug=True)
