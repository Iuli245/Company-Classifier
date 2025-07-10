import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

print("🔄 Loading data...")
company_df = pd.read_csv("output/company_profiles_debug.csv")
taxonomy_df = pd.read_excel("data/insurance_taxonomy.xlsx")
taxonomy_df['label_clean'] = taxonomy_df['label'].astype(str).str.strip()
taxonomy_labels = taxonomy_df['label_clean'].tolist()

print("📦 Loading model: all-mpnet-base-v2")
model = SentenceTransformer("all-mpnet-base-v2")

print("🧠 Encoding embeddings...")
company_embeddings = model.encode(company_df['company_profile'].tolist(), convert_to_tensor=True, show_progress_bar=True)
taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True)

print("📊 Calculating cosine similarity...")
similarity_scores = util.cos_sim(company_embeddings, taxonomy_embeddings)

# === NOU: salvăm și sursa + scor ===
predicted_labels = []
label_sources = []
label_scores = []

def get_label_info(sim_scores_row, labels, high_thresh=0.6, mid_thresh=0.45):
    strong = [(i, score) for i, score in enumerate(sim_scores_row) if score >= high_thresh]
    if strong:
        idxs = [i for i, _ in strong]
        scores = [float(sim_scores_row[i]) for i in idxs]
        return [labels[i] for i in idxs], "direct", max(scores)

    fallback1 = [(i, score) for i, score in enumerate(sim_scores_row) if mid_thresh <= score < high_thresh]
    if fallback1:
        idxs = [i for i, _ in fallback1]
        scores = [float(sim_scores_row[i]) for i in idxs]
        return [labels[i] for i in idxs], "fallback1", max(scores)

    max_index = sim_scores_row.argmax().item()
    max_score = float(sim_scores_row[max_index])
    return [labels[max_index]], "fallback2", max_score

print("🔁 Generating final labels with sources and scores...")
for row in similarity_scores:
    labels, source, score = get_label_info(row, taxonomy_labels)
    predicted_labels.append(labels)
    label_sources.append(source)
    label_scores.append(score)

# === Adaugă coloane noi ===
company_df['insurance_label'] = predicted_labels
company_df['label_source'] = label_sources
company_df['label_score'] = label_scores

output_path = "output/ml_insurance_challenge_labeled_mpnet_final_valori.csv"
os.makedirs("output", exist_ok=True)
company_df.to_csv(output_path, index=False)

print(f"✅ Clasificare completă. Fișierul final cu etichete, sursă și scor a fost salvat în: {output_path}")
