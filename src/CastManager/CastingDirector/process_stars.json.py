import pandas as pd
import json
from collections import defaultdict

# Load original dataset
df = pd.read_csv("CastManager\CastingDirector\model\imdb_top_1000.csv")
df = df[['Star1', 'Star2', 'Star3', 'Star4', 'Director']]
df.dropna(inplace=True)

# Load existing stars.json
with open("CastManager/CastingDirector/static/stars.json", "r") as f:
    stars = json.load(f)

# Build a role lookup
role_lookup = defaultdict(set)
for role in ['Star1', 'Star2', 'Star3', 'Star4']:
    for name in df[role]:
        role_lookup[name].add("actor")
for name in df['Director']:
    role_lookup[name].add("director")

# Add labels
for star in stars:
    name = star['name']
    labels = role_lookup.get(name, set())
    # If someone is both actor and director, list both
    star['label'] = "/".join(sorted(labels)).title() if labels else "Unknown"

# Save updated stars.json
with open("CastManager/CastingDirector/static/stars.json", "w") as f:
    json.dump(stars, f, indent=2)

print("✅ stars.json updated with labels.")
