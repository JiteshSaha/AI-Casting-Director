import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model import CastRatingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

df = pd.read_csv("./model/imdb_top_1000.csv")

print(df.columns)

df = df[['Star1', 'Star2', 'Star3', 'Star4', 'Director', 'Meta_score']]
# Drop rows with missing values
df.dropna(inplace=True)


df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
df = df.dropna(subset=['Meta_score'])



# Combine all star names into one list to encode uniquely
all_stars = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4'] , df['Director']])
label_encoder = LabelEncoder()
label_encoder.fit(all_stars)

# Save during training
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


# Encode cast members
for col in ['Star1', 'Star2', 'Star3', 'Star4', 'Director']:
    df[col] = label_encoder.transform(df[col])

# Convert to numpy arrays for model input
X_cast = df[['Star1', 'Star2', 'Star3', 'Star4', 'Director']].values.astype(np.int64)
y_rating = df['Meta_score'].values.astype(np.float32)

print(f"Sample X: {X_cast[:3]}")
print(f"Sample y: {y_rating[:3]}")
print(f"Number of unique cast members: {len(label_encoder.classes_)}")

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_cast, y_rating, test_size=0.2, random_state=42)

class MovieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MovieDataset(X_train, y_train)
val_ds = MovieDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'{device=}')


model = CastRatingRegressor(num_people=len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")



def predict_rating(cast_names):
    # cast_names: list of 4 actors + 1 director
    ids = [label_encoder.transform([name])[0] if name in label_encoder.classes_ else 0 for name in cast_names]
    x = torch.tensor([ids], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        rating = model(x).item()
    return round(rating, 2)



rating = predict_rating(["Leonardo DiCaprio", "Tom Hanks", "Brad Pitt", "Morgan Freeman", "Christopher Nolan"])



print(rating)


from safetensors.torch import save_file

# Save model weights
save_file(model.state_dict(), "./model/cast_rating_model.safetensors")
