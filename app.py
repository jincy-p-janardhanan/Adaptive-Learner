from flask import Flask, request, render_template, jsonify, send_file
import json
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# S3 Configuration
BUCKET_NAME = "adaptive-learner-bucket"
JSON_FILE_KEY = "dataset.json"  # The key name of your JSON file in S3

# Initialize the S3 client
s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION'), 
                         aws_access_key_id=os.getenv('ACCESS_KEY_ID'), 
                         aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'))

# Load the JSON file from S3 if it exists, otherwise create a default structure
def load_data_from_s3():
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=JSON_FILE_KEY)
        data = json.load(response['Body'])
        return data
    except s3_client.exceptions.NoSuchKey:
        # If file does not exist, return an empty list
        return []

def save_data_to_s3(data):
    json_content = json.dumps(data, indent=4)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=JSON_FILE_KEY, Body=json_content, ContentType='application/json')

# Initialize the data
data = load_data_from_s3()

@app.route('/')
def index():
    return render_template('data-entry.html', data=data)

@app.route('/add_entry', methods=['POST'])
def add_entry():
    # Retrieve form data
    difficulty = request.form.get("difficulty", type=int)
    title = request.form.get("title")
    type_ = request.form.get("type")
    prologue = request.form.get("prologue")
    content = request.form.get("content")
    new_words = json.loads(request.form.get("new_words"))
    exercises = json.loads(request.form.get("exercises"))

    # Automatically assign an ID
    next_id = max((entry['id'] for entry in data), default=0) + 1

    # Add new entry
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

    # Save to JSON file on S3
    save_data_to_s3(data)

    return jsonify({"success": True, "message": "Entry added successfully!"})

@app.route('/get_chapter', methods=['GET'])
def get_chapter():
    search_query = request.args.get('query', '').strip()
    if not search_query:
        return jsonify({"success": False, "message": "Query cannot be empty"}), 400
    
    # Search for chapters by id or title
    for chapter in data:
        if str(chapter['id']) == search_query or chapter['title'].lower() == search_query.lower():
            return jsonify({"success": True, "chapter": chapter}), 200

    return jsonify({"success": False, "message": "Chapter not found"}), 404

@app.route('/update_entry', methods=['POST'])
def update_entry():
    chapter_id = request.form.get("id", type=int)
    if not chapter_id:
        return jsonify({"success": False, "message": "Chapter ID is required"}), 400

    # Retrieve form data
    difficulty = request.form.get("difficulty", type=int)
    title = request.form.get("title")
    type_ = request.form.get("type")
    prologue = request.form.get("prologue")
    content = request.form.get("content")
    new_words = json.loads(request.form.get("new_words"))
    exercises = json.loads(request.form.get("exercises"))

    # Find and update the chapter
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

            # Save to JSON file on S3
            save_data_to_s3(data)

            return jsonify({"success": True, "message": "Chapter updated successfully!"}), 200

    return jsonify({"success": False, "message": "Chapter not found!"}), 404

@app.route('/dataset', methods=['GET'])
def download_dataset():
    try:
        # Download the JSON file from S3 to a local BytesIO object
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=JSON_FILE_KEY)
        json_data = response['Body'].read()

        # Send the file as an attachment
        return send_file(BytesIO(json_data), as_attachment=True, download_name=JSON_FILE_KEY, mimetype='application/json')
    except s3_client.exceptions.NoSuchKey:
        return jsonify({"message": "Dataset not found!"}), 404

if __name__ == "__main__":
    app.run(debug=True)
