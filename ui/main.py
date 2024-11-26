from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

from gpt4all import GPT4All
gpt4allmodel = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf") 

class Model:
    def run(self, current_level, question_level):
        global gpt4allmodel

        items = ["poem", "story", "drama"]
        content_prompt = (f"This should have one {random.choice(items)} "
                          f"of difficulty level {current_level} out of 10 for a beginner English learner. "
                          "If it's a poem, format it clearly with each line separated by a newline.")

        prompt = f"""
Generate a response with the following format:

Content:
{content_prompt}

Question:
Include one {question_level} question based on the content.

Answer:
Include the answer to the question.
"""

        response = gpt4allmodel.generate(prompt, max_tokens=1024)

        # Parse the response using simple string operations
        try:
            # Split the response into sections
            content = self.extract_section(response, "Content:", "Question:")
            question = self.extract_section(response, "Question:", "Answer:")
            answer = self.extract_section(response, "Answer:")

            print(content.strip())
            print(question.strip())
            print(answer.strip())

            return {
                "content": content.strip(),
                "questions": question.strip(),
                "correct_answer": answer.strip(),
                "Level": current_level
            }

        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response text was: {response}")
            return {
                "content": "Failed to generate content.",
                "questions": "No question generated.",
                "correct_answer": "No answer available.",
                "Level": current_level
            }

    def extract_section(self, response, start_key, end_key=None):
        """Extracts a section from the response between start_key and end_key."""
        start_index = response.find(start_key)
        if start_index == -1:
            return ""
        start_index += len(start_key)
        if end_key:
            end_index = response.find(end_key, start_index)
            if end_index == -1:
                return response[start_index:]
            return response[start_index:end_index]
        return response[start_index:]

    def check(self, user_answer, correct_answer):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([correct_answer, user_answer])

        cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return cos_sim[0][0]

model = Model()

state = {
    "current_level": 1,
    "final_score": None,
    "output": None
}

question_levels = ["easy", "medium", "hard", "very hard"]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/quiz', methods=['GET'])
def quiz():
    level = request.args.get('level', default=1, type=int)
    if level != state["current_level"]:
        state["current_level"] = level
    print(level)
    difficulty_index = 0
    state["current_level"] = level
    state["difficulty_index"] = difficulty_index
    state["output"] = model.run(level, question_levels[difficulty_index])
    
    return render_template('quiz.html', output=state["output"])

@app.route('/submit', methods=['POST'])
def submit():
    global state
    user_answer = request.form['user_answer']
    correct_answer = state["output"]["correct_answer"]
    is_correct = model.check(user_answer, correct_answer)

    leveled_up = False
    
    if is_correct >= 0.3:
        # Move to the next difficulty level if the answer is correct
        state["difficulty_index"] += 1
        
        if state["difficulty_index"] >= len(question_levels):
            # If all difficulty levels are completed, level up
            state["difficulty_index"] = 0  # Reset difficulty to easy for the new level
            state["current_level"] += 1
            leveled_up = True
        else:
            # Continue with the next difficulty level
            difficulty_level = question_levels[state["difficulty_index"]]
            state["output"] = model.run(state["current_level"], difficulty_level)
    else:
        return redirect("/learn")
    
    # Prepare a response indicating if the user leveled up or not
    return render_template('quiz.html', output=state["output"], 
                           leveled_up = leveled_up, new_level = state["current_level"], 
                           current_difficulty = question_levels[state["difficulty_index"]])


@app.route('/learn', methods=['GET'])
def learn():
    level = request.args.get('level', default=1, type=int)
    if level != state["current_level"]:
        state["current_level"] = level
        
    global gpt4allmodel
    items = ["poem", "story", "drama"]
    choice = {random.choice(items)}
    content_prompt = (f"Generate a response in the the following format. No need of any styling:"
                        "Title: title of the {choice}"
                        "{choice}: Generate one  {choice} of difficulty level {level} out of 10 for a beginner English learner. If it's a poem, format it clearly with each line separated by a newline."
                        )
    response = gpt4allmodel.generate(content_prompt, max_tokens=1024)
    title = model.extract_section(response, "Title:", f"{choice}")
    content = model.extract_section(response, f"{choice}")

    return render_template('learn.html', title=title, content = content, current_level = state["current_level"])

if __name__ == '__main__':
    app.run(debug=True)
