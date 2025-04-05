import torch
import pandas as pd
from safetensors.torch import load_file
from sklearn.preprocessing import LabelEncoder
from model import CastRatingRegressor  # You can copy the model class to a separate model.py
import pickle

# === Load label encoder ===
df = pd.read_csv("./model/imdb_top_1000.csv")
df = df[['Star1', 'Star2', 'Star3', 'Star4', 'Director']].dropna()

all_names = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4'], df['Director']])

# Then during prediction, load it:
with open("./model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_people = len(label_encoder.classes_)

model = CastRatingRegressor(num_people=num_people)
model.load_state_dict(load_file("./model/cast_rating_model.safetensors"))
model.to(device)
model.eval()

# === Get input from user ===
print("Enter cast and director for prediction:")
star1 = input("Star 1: ")
star2 = input("Star 2: ")
star3 = input("Star 3: ")
star4 = input("Star 4: ")
director = input("Director: ")

names = [star1, star2, star3, star4, director]
ids = []
for name in names:
    if name in label_encoder.classes_:
        ids.append(label_encoder.transform([name])[0])
    else:
        print(f"Warning: '{name}' not in training data. Using fallback ID 0.")
        ids.append(0)  # fallback if unknown

x = torch.tensor([ids], dtype=torch.long).to(device)

# === Predict ===
with torch.no_grad():
    pred = model(x).item()

print(f"\n🎬 Predicted Meta Score for {', '.join(names)}: {round(pred, 2)}")