from data_loader import load_data

df = load_data()
print(df[['title', 'text', 'label']].head())