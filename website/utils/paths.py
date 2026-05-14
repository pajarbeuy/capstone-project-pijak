from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = BASE_DIR / "model" / "model_svm_best.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "vibesight_ml_ready_v1.0.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "vibesight_train_v1.0.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "vibesight_test_v1.0.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_v1.0_20260502_2121.json"
