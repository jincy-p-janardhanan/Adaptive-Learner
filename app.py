from flask import Flask, request, render_template, jsonify
import json
import os

app = Flask(__name__)

# Load the JSON file if it exists, otherwise create a default structure
JSON_FILE = "static/dataset.json"

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as file:
        data = json.load(file)
else:
    data = []

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

    # Save to JSON file
    with open(JSON_FILE, "w") as file:
        json.dump(data, file, indent=4)

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

            # Save to JSON file
            with open(JSON_FILE, "w") as file:
                json.dump(data, file, indent=4)

            return jsonify({"success": True, "message": "Chapter updated successfully!"}), 200

    return jsonify({"success": False, "message": "Chapter not found!"}), 404


if __name__ == "__main__":
    app.run(debug=True)
