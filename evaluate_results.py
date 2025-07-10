import pandas as pd

df = pd.read_csv("output/ml_insurance_challenge_labeled.csv")

# Alege 10 companii cu descriere clară și etichete atribuite
sample = df[df['insurance_label'].apply(lambda x: len(eval(x)) > 0)].sample(10, random_state=42)

# Salvează pentru inspecție
sample.to_csv("output/validation_sample.csv", index=False)
