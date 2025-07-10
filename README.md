🧩Understanding the Task and Exploring the Data
At the beginning of the project, the classification task was not strictly defined as a supervised learning problem. Instead, we were provided with:

-A CSV file (ml_insurance_challenge.csv) containing company-level information such as description, tags, sector, category, and niche
-A static insurance taxonomy (Excel file) representing potential target labels

We inferred that the goal was to assign one or more relevant taxonomy labels to each company based on semantic similarity between their profile and label descriptions.

To better understand the data, we:

-Inspected company fields for completeness and relevance
-Identified missing or noisy values in tags and text descriptions
-Chose to create a new column full_profile combining all informative fields into one coherent input for embedding models

This step was crucial to ensure the model had sufficient context to make accurate predictions.

🔧 Project Stages

1. Data Preprocessing

-Initial file: ml_insurance_challenge.csv.

-Cleaning and concatenating relevant information into a company_profile field, which became the foundation for embeddings.

-Saved intermediate file: company_profiles_debug.csv.

2. Embedding Generation

-Initial model used: all-MiniLM-L6-v2.

-Later tested other models: paraphrase-MiniLM-L12-v2, multi-qa-MiniLM-L6-cos-v1, all-mpnet-base-v2.

-Method: cos_sim between company description and each taxonomy label (embeddings generated separately).

3. Label Assignment Based on Similarity Score

-Initially used a single threshold (threshold=0.45) and retained all labels above this score.

-Later introduced fallback mechanisms for companies without labels:

-Top 1 label assigned if the threshold was not met.

4. Progressive Improvements

-Analyzed companies without labels and adjusted the threshold.

-Evaluated models by comparing the number of labeled companies per model.

-Added extra context: business_tags, sector, category, niche to the complete profile field (full_profile).

-New input file: company_profiles_all_tags_debug.csv.

5. Advanced Multi-Tier Fallback Logic

Implemented 4-level fallback logic:

-score >= 0.6 → direct labels.

-0.45 <= score < 0.6 → top 3 labels.

-0.3 <= score < 0.45 → top 2 labels.

-otherwise → top 1 label (most similar).

-Added additional columns: similarity_strategy and max_similarity_score for transparency.

6. Results Analysis

-Achieved 100% labeling due to fallback strategies.

-Manually validated cases with low scores (< 0.45).

-In general, about 70–80% of generated labels were reasonable, some highly relevant.

📦 Final Output

Result file: ml_insurance_challenge_labeled_mpnet_extended_toate_campurile_fallback_final.csv

Key columns:

-insurance_label – assigned labels.

-similarity_strategy – strategy used (direct or fallback).

-max_similarity_score – highest score per company.

📌 Conclusions

-Using a powerful model (all-mpnet-base-v2) and extending context in full_profile brought significant improvements.

-The multi-tier fallback strategy was essential in ensuring complete dataset coverage.

🌱 Recommendations for Future Extensions

-Fine-tuning on domain-specific data: Training a custom model on the insurance taxonomy labels may boost accuracy.

-Supervised classification: Implement a multi-label classification approach using embeddings + ML models (e.g., logistic regression, XGBoost).

-Human-assisted validation: Develop a semi-automated system for expert review of low-score or fallback predictions.

-Additional data sources: Enhance full_profile with website content, extended descriptions, or relevant financial data.

-Quantitative evaluation: Manually label a subset of the data to calculate precision, recall, and F1 scores.
