import pandas as pd
import ast


# Încarcă datele de input
company_df = pd.read_csv("data/ml_insurance_challenge.csv")
taxonomy_df = pd.read_excel("data/insurance_taxonomy.xlsx")


#curat tagurile
def clean_tags(tag_str):
    try:
        return ', '.join(ast.literal_eval(tag_str))
    except (ValueError, SyntaxError):
        return ''


company_df['clean_tags'] = company_df['business_tags'].apply(clean_tags)

#costruiesc company_profile
def build_company_profile(row):
    return f"{row['description']} | Tags: {row['clean_tags']} | Sector: {row['sector']} | Category: {row['category']} | Niche: {row['niche']}"

company_df['company_profile'] = company_df.apply(build_company_profile, axis=1)

#curat etichetele din taxonomy
taxonomy_df['label_clean'] = taxonomy_df['label'].astype(str).str.strip()
taxonomy_labels = taxonomy_df['label_clean'].tolist()


#creez un fisier intermediar cu datele curatate
company_df[['company_profile']].to_csv("output/company_profiles_debug.csv", index=False)
