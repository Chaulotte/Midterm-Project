import pickle
from flask import Flask, request, jsonify

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("penguin-sex-prediction")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    penguin = request.get_json()
    if not penguin:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # Normalize categorical strings to match training preprocessing
    clean_record = {}
    for k, v in penguin.items():
        if isinstance(v, str):
            clean_record[k.lower()] = v.replace(" ", "_").lower()
        else:
            clean_record[k.lower()] = v

    X = dv.transform([clean_record])
    prob_male = float(model.predict_proba(X)[0, 1])
    is_male = bool(prob_male >= 0.5)

    result = {
        "male_probability": round(prob_male, 4),
        "predicted_sex": "male" if is_male else "female",
        "is_male": is_male,
    }
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
