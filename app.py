# app.py
from flask import Flask, render_template, request
from flask import jsonify
import json
from model import ReviewClassifier

review_classifier = ReviewClassifier(model_path="model/review_classifier/decoder/mamba_model/binary_classification_decoder.pt",
                                      encoder_path="model/review_classifier/embedder/nomic_model")
app = Flask(__name__, template_folder='templates')


# Display your index page


@app.route("/")
def index():
    return render_template("/index.html")
# A function to add two numbers


@app.route("/api/<batch>", methods=["POST"])
def analyse(batch: str):
    if batch == "analyze-batch":
        payload = request.get_json()
        reviews = [item["review"] for item in payload]
        locations = [item["location"] for item in payload]

    else:
        review = request.args.get("review")
        location = request.args.get("location")
        reviews = [review]
        locations = [location]
    labels, confidences = review_classifier.classify(reviews, locations)
    result = []
    for i in range(len(reviews)):
        result.append(
            {
                "review": reviews[i],
                "confidence": confidences[i],
                "label": labels[i]
            }
        )
    return json.dumps(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
