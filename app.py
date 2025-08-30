# app.py
from flask import Flask, render_template, request
from flask import jsonify
import json
from model import ReviewClassifier

review_classifier = ReviewClassifier(
    model_path=r"model\review_classifier\decoder\mamba_model\multi_classification_decoder.pt", 
    encoder_path="model/review_classifier/embedder/nomic_model", 
    multi_classes=True
)
app = Flask(__name__, template_folder='templates')


# Display your index page
@app.route("/")
def index():
    return render_template("/index.html")
# A function to add two numbers
@app.route("/api/<batch>", methods=["POST"])
def analyse(batch: str):
    if batch == "analyze-batch":
        data = request.get_json()
        print(data)
        
    else:
        review = request.args.get("review")
        location = request.args.get("location")
        data = [[review]]
    # result = model.analyse(data)
    result = """[
  {
    "location": "macs",
    "review": "Not that good",
    "cf": 0.1,
    "label": 1
  },
  {
    "location": "macs",
    "review": "that",
    "cf": 0.645,
    "label": 0
  },
  {
    "location": "macs",
    "review": "good",
    "cf": 0.1,
    "label": 1
  },
  {
    "location": "macs",
    "review": "Not it",
    "cf": 0.4,
    "label": 0
  },
  {
    "location": "macs",
    "review": "Not it good",
    "cf": 0.6,
    "label": 0
  },
  {
    "location": "macs",
    "review": "Not that",
    "cf": 0.9,
    "label": 0
  },
  {
    "location": "macs",
    "review": "Not",
    "cf": 0.04933643634664364,
    "label": 1
  }
]"""
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)