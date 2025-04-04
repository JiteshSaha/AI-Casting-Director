import pickle

def get_label_encoder(csv_path):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path)
    df = df[['Star1', 'Star2', 'Star3', 'Star4', 'Director']].dropna()
    all_names = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4'], df['Director']])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_names)

    # Save encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    return label_encoder


def encode_cast_input(names, label_encoder):
    ids = []
    for name in names:
        if name in label_encoder.classes_:
            ids.append(label_encoder.transform([name])[0])
        else:
            print(f"Warning: '{name}' not in training data. Using fallback ID 0.")
            ids.append(0)
    return ids
