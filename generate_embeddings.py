import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# === 1. Încarcă fișierele preprocesate ===
print("📦 Loading preprocessed data...")
company_df = pd.read_csv("output/company_profiles_debug.csv")
taxonomy_df = pd.read_excel("data/insurance_taxonomy.xlsx")

# === 2. Curățare labeluri taxonomy ===
taxonomy_df['label_clean'] = taxonomy_df['label'].astype(str).str.strip()
taxonomy_labels = taxonomy_df['label_clean'].tolist()

# Verifică existența coloanei 'company_profile'
if 'company_profile' not in company_df.columns:
    raise ValueError("Coloana 'company_profile' nu există în fișierul preprocesat. Verifică preprocess_data.py.")

# === 3. Încarcă modelul de embeddings ===
print("🧠 Loading sentence-transformer model...")
model = SentenceTransformer('all-mpnet-base-v2')

# === 4. Generează embeddings ===
print("🧬 Generating embeddings for companies and labels...")
company_embeddings = model.encode(company_df['company_profile'].tolist(), convert_to_tensor=True, show_progress_bar=True)
taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True)

# === 5. Calculează similaritatea cosine ===
print("📐 Computing cosine similarities...")
similarity_scores = util.cos_sim(company_embeddings, taxonomy_embeddings)

# === 6. Etichete cu fallback dacă niciuna nu trece pragul ===
def get_labels_with_fallback(sim_scores_row, labels, threshold=0.45):
    passing = [(i, score) for i, score in enumerate(sim_scores_row) if score >= threshold]
    if passing:
        return [labels[i] for i, _ in passing]
    else:
        max_index = sim_scores_row.argmax().item()
        return [labels[max_index]]

predicted_labels = [get_labels_with_fallback(row, taxonomy_labels) for row in similarity_scores]

# === 7. Adaugă etichetele în dataframe ===
company_df['insurance_label'] = predicted_labels

# === 8. Salvează rezultatul final ===
output_path = "output/ml_insurance_challenge_labeled_mpnet_fallback.csv"
os.makedirs("output", exist_ok=True)
company_df.to_csv(output_path, index=False)

print(f"✅ Clasificare completă cu fallback. Fișierul este salvat în: {output_path}")
