import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# === MODELE DE TESTAT === (de la rapid la lent)
MODELS_TO_TEST = [
    "all-MiniLM-L6-v2",                 # foarte rapid, model mic
    "multi-qa-MiniLM-L6-cos-v1",        # rapid, optimizat pentru QA
    "paraphrase-MiniLM-L12-v2",         # mediu, balansat
    "all-mpnet-base-v2",                # cel mai lent, dar mai performant
]

# === Încărcare date preprocesate ===
company_df_raw = pd.read_csv("output/company_profiles_debug.csv")
taxonomy_df = pd.read_excel("data/insurance_taxonomy.xlsx")
taxonomy_df['label_clean'] = taxonomy_df['label'].astype(str).str.strip()
taxonomy_labels = taxonomy_df['label_clean'].tolist()

# === Funcție de extragere etichete fără fallback ===
def get_labels_no_fallback(sim_scores_row, labels, threshold=0.45):
    return [labels[i] for i, score in enumerate(sim_scores_row) if score >= threshold]

# === Funcție de verificare câte etichete sunt goale ===
def count_empty(predictions):
    return sum(len(label) == 0 for label in predictions)

# === Rulează clasificarea pentru fiecare model ===
for model_name in MODELS_TO_TEST:
    print(f"\n🚀 Running with model: {model_name}")

    # 1. Încarcă modelul
    model = SentenceTransformer(model_name)

    # 2. Creează embeddings
    company_embeddings = model.encode(
        company_df_raw['company_profile'].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    taxonomy_embeddings = model.encode(
        taxonomy_labels,
        convert_to_tensor=True
    )

    # 3. Calculează similaritatea cosine
    similarity_scores = util.cos_sim(company_embeddings, taxonomy_embeddings)

    # 4. Aplică filtrul de etichete (fără fallback)
    predicted_labels = [get_labels_no_fallback(row, taxonomy_labels) for row in similarity_scores]

    # 5. Creează DataFrame cu rezultatul
    company_df = company_df_raw.copy()
    company_df['insurance_label'] = predicted_labels

    # 6. Verifică etichetele goale
    empty_count = count_empty(predicted_labels)
    print(f"📊 Model '{model_name}': {empty_count} companii fără etichete.")

    # 7. Salvează fișierul
    output_file = f"output/ml_insurance_challenge_labeled_{model_name.replace('/', '_')}.csv"
    os.makedirs("output", exist_ok=True)
    company_df.to_csv(output_file, index=False)
    print(f"💾 Fișier salvat: {output_file}")
