from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

MODEL_PATH = "model.pkl"

model = joblib.load(MODEL_PATH)


@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    text = data.get("text")
    threshold = float(data.get("threshold", 0.5))

    prediction, propensity = score(text, model, threshold)

    return jsonify({
        "prediction": int(prediction),
        "propensity": propensity
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)