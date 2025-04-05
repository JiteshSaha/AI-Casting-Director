from flask import Flask, request, render_template, jsonify
from model.model import CastRatingRegressor
from safetensors.torch import load_file
import pickle, torch
from model.utils import get_label_encoder, encode_cast_input

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & label encoder once
with open(r"./model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = CastRatingRegressor(num_people=len(label_encoder.classes_))
model.load_state_dict(load_file(r"./model/cast_rating_model.safetensors"))
model.to(device)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    names = data.get("names", [])
    try:
        ids = encode_cast_input(names, label_encoder)
        x = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            pred = model(x).item()
        return jsonify({"rating": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
