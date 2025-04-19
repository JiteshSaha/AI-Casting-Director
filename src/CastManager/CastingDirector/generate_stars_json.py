from tmdbv3api import TMDb, Person
import pickle, json

# Load label encoder
with open("CastingDirector/model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# TMDb setup
tmdb = TMDb()
tmdb.api_key = "3c9ee9846a271429c23732d4da9f6bc1"


tmdb.language = "en"

person = Person()
names = label_encoder.classes_

# Placeholder if no image found
default_img = "/CastingDirector/static/default.jpg"
stars_data = []

for name in names:

    print(f'{name=}')
    try:
        result = person.search(name)
        if result:
            img_path = result[0].profile_path
            image_url = f"https://image.tmdb.org/t/p/w185{img_path}" if img_path else default_img
        else:
            image_url = default_img 
    except Exception as e:
        print(f"Error fetching {name}: {e}")
        image_url = default_img

    stars_data.append({"name": name, "image": image_url})

# Save stars.json
with open("CastingDirector/static/stars.json", "w") as f:
    json.dump(stars_data, f, indent=2)

print(f"✅ stars.json created with {len(stars_data)} entries.")
