# app.py
from flask import Flask, render_template, request
from flask import jsonify
import json

app = Flask(__name__, template_folder='templates')

# Display your index page
@app.route("/")
def index():
    return render_template("/index.html")
# A function to add two numbers
@app.route("/model/<batch>", methods=["POST"])
def analyse(batch: str):
    if batch == "batch":
        data = request.get_json()
        
    else:
        review = request.args.get("review")
        location = request.args.get("location")
        data = [[review]]
    # result = model.analyse(data)
    result = {}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)