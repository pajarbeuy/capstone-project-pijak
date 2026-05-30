from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent

# Model & Vectorizer paths - menggunakan local copies dalam dashboard-streamlit folder
# Ini memastikan path bekerja di Streamlit Cloud
MODEL_PATH = BASE_DIR / "model" / "model_svm_best.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

# Dataset paths
DATA_PATH = PROJECT_ROOT / "dataset" / "tokopedia_reviews.csv"
TRAIN_PATH = PROJECT_ROOT / "dataset" / "tokopedia_reviews.csv"
TEST_PATH = PROJECT_ROOT / "dataset" / "tokopedia_reviews.csv"
METADATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "metadata_v1.0_20260502_2121.json"
