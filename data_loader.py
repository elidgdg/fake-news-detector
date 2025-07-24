import pandas as pd

def load_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(real_path)

    # Label: 0 = fake, 1 = real
    df_fake["label"] = 0
    df_true["label"] = 1

    # Combine and shuffle
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)