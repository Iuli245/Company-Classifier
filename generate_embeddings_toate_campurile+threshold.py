import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import matplotlib.pyplot as plt

# === 1. Încarcă datele ===
print("📥 Loading data...")
company_df = pd.read_csv("output/company_profiles_all_tags_debug.csv")
taxonomy_df = pd.read_excel("data/insurance_taxonomy.xlsx")
taxonomy_df['label_clean'] = taxonomy_df['label'].astype(str).str.strip()
taxonomy_labels = taxonomy_df['label_clean'].tolist()

# === 2. Creează profil extins cu context suplimentar ===
print("🧠 Generating full profiles with extra context (business_tags, sector, category, niche)...")
company_df["full_profile"] = (
    company_df["company_profile"].fillna('') + ". " +
    "Tags: " + company_df["business_tags"].fillna('') + ". " +
    "Sector: " + company_df["sector"].fillna('') + ". " +
    "Category: " + company_df["category"].fillna('') + ". " +
    "Niche: " + company_df["niche"].fillna('')
)

# === 3. Încarcă modelul ===
print("⚙️ Loading model: all-mpnet-base-v2...")
model = SentenceTransformer("all-mpnet-base-v2")

# === 4. Generează embeddings ===
print("🔎 Generating embeddings...")
company_embeddings = model.encode(company_df["full_profile"].tolist(), convert_to_tensor=True, show_progress_bar=True)
taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True)

# === 5. Calculează similarități cosine ===
print("📏 Calculating cosine similarities...")
similarity_scores = util.cos_sim(company_embeddings, taxonomy_embeddings)

# === 6. Funcție fallback multi-nivel cu scor și strategie ===
def get_dynamic_fallback_with_strategy(sim_row, all_labels):
    scored_labels = list(enumerate(sim_row))
    scored_labels.sort(key=lambda x: x[1], reverse=True)
    max_sim_score = float(scored_labels[0][1])

    if max_sim_score >= 0.6:
        strategy = "primary"
        selected_labels = [all_labels[i] for i, s in scored_labels if s >= 0.6]
    elif max_sim_score >= 0.45:
        strategy = "top3"
        selected_labels = [all_labels[i] for i, s in scored_labels if 0.45 <= s < 0.6][:3]
    elif max_sim_score >= 0.3:
        strategy = "top2"
        selected_labels = [all_labels[i] for i, s in scored_labels if 0.3 <= s < 0.45][:2]
    else:
        strategy = "top1"
        selected_labels = [all_labels[scored_labels[0][0]]]

    return selected_labels, strategy, max_sim_score

# === 7. Aplică fallback și extrage scoruri și strategii ===
print("🧪 Applying multi-tier fallback logic...")
results = [get_dynamic_fallback_with_strategy(row.tolist(), taxonomy_labels) for row in similarity_scores]
company_df["insurance_label"] = [r[0] for r in results]
company_df["fallback_strategy"] = [r[1] for r in results]
company_df["similarity_score"] = [r[2] for r in results]

# === 8. Statistici generale ===
empty_count = sum(len(labels) == 0 for labels in company_df["insurance_label"])
non_empty_count = len(company_df) - empty_count

print(f"\n📊 Summary with dynamic fallback:")
print(f" - Etichetate: {non_empty_count}")
print(f" - Neetichetate: {empty_count} (ar trebui 0)")

# === 9. Grafic distribuție ===
plt.figure(figsize=(6, 4))
plt.bar(["Etichetate", "Neetichetate"], [non_empty_count, empty_count], color=["green", "red"])
plt.title("Clasificare cu fallback multiplu (profil extins)")
plt.ylabel("Număr companii")
plt.tight_layout()
plt.show()

# === 10. Salvare fișier final ===
output_path = "output/ml_insurance_challenge_labeled_mpnet_extended_toate_campurile_fallback_final.csv"
os.makedirs("output", exist_ok=True)
company_df.to_csv(output_path, index=False)
print(f"\n✅ Fișier salvat în: {output_path}")
