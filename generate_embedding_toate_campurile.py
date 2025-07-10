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

# === 2. Creează profil extins cu business_tags, sector, category și niche ===
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

# === 6. Aplică threshold fără fallback ===
THRESHOLD = 0.45
print(f"🧪 Applying threshold: {THRESHOLD} (no fallback)")
predicted_labels = [
    [taxonomy_labels[i] for i, score in enumerate(row) if score >= THRESHOLD]
    for row in similarity_scores
]

# === 7. Adaugă etichete în DataFrame ===
company_df["insurance_label"] = predicted_labels

# === 8. Analizează câte rânduri sunt goale ===
empty_count = sum(len(labels) == 0 for labels in predicted_labels)
non_empty_count = len(predicted_labels) - empty_count

print(f"\n📊 Summary:")
print(f" - Etichetate (>= {THRESHOLD}): {non_empty_count}")
print(f" - Neetichetate: {empty_count}")

# === 9. Grafic bară distribuție ===
plt.figure(figsize=(6, 4))
plt.bar(["Etichetate", "Neetichetate"], [non_empty_count, empty_count], color=["green", "red"])
plt.title("Clasificare fără fallback (profil extins: tags, sector, category, niche)")
plt.ylabel("Număr companii")
plt.tight_layout()
plt.show()

# === 10. Salvare fișier final ===
output_path = "output/ml_insurance_challenge_labeled_mpnet_extended_toate_campurile.csv"
os.makedirs("output", exist_ok=True)
company_df.to_csv(output_path, index=False)
print(f"\n✅ Fișier salvat în: {output_path}")
