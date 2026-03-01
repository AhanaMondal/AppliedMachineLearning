from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

MODEL_PATH = r"C:\Users\Ahana\Documents\AppliedMachineLearning\AML_Assignment2\notebooks\mlruns\302143816419166244\models\m-c24d00be04c04ce39af7bda701c7ca18\artifacts\model.pkl"

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
    app.run(port=5000)