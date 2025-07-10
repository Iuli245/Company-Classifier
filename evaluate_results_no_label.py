import pandas as pd
import ast

# Încarcă fișierul generat de mpnet
df = pd.read_csv("output/ml_insurance_challenge_labeled.csv")  # sau calea corectă

# Funcție pentru a detecta liste goale
def is_empty_list(label_str):
    try:
        return len(ast.literal_eval(label_str)) == 0
    except:
        return True

# Filtrare companii fără etichete
no_label_df = df[df['insurance_label'].apply(is_empty_list)]

# Eșantion pentru inspecție manuală
sample = no_label_df.sample(10, random_state=42)

# Salvare în fișier de validare
sample.to_csv("output/validation_sample_no_labels.csv", index=False)
