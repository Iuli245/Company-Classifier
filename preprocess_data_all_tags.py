import pandas as pd
import os

# === 1. Încarcă fișierul brut ===
print("📥 Loading raw company list...")
input_path = "data/ml_insurance_challenge.csv"
df = pd.read_csv(input_path)

# === 2. Creează coloana de profil principal (doar description) ===
print("🧠 Generating 'company_profile' based on description...")
df["company_profile"] = df["description"].fillna("")

# === 3. Asigură-te că celelalte câmpuri sunt păstrate și procesate corect ===
for col in ["business_tags", "sector", "category", "niche"]:
    if col not in df.columns:
        print(f"⚠️ Warning: Missing expected column: {col}")
        df[col] = ""

# === 4. Creează directorul de output dacă nu există ===
os.makedirs("output", exist_ok=True)

# === 5. Salvează fișierul preprocesat ===
output_path = "output/company_profiles_all_tags_debug.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Preprocessing complete. File saved to: {output_path}")
