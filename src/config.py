import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = DATA_DIR / "vector_store"

for folder in [RAW_DIR, PROCESSED_DIR, VECTOR_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

NCBI_TOOL = "RareDxRAG"

HPOA_URL = (
    "https://github.com/obophenotype/"
    "human-phenotype-ontology/releases/latest/download/"
    "phenotype.hpoa"
)

HPO_OBO_URL = (
    "https://github.com/obophenotype/"
    "human-phenotype-ontology/releases/latest/download/"
    "hp.obo"
)

GENES_TO_DISEASE_URL = (
    "https://github.com/obophenotype/"
    "human-phenotype-ontology/releases/latest/download/"
    "genes_to_disease.txt"
)

HPOA_FILE = RAW_DIR / "phenotype.hpoa"
HPO_OBO_FILE = RAW_DIR / "hp.obo"
GENES_FILE = RAW_DIR / "genes_to_disease.txt"

DISEASE_DATASET = PROCESSED_DIR / "diseases.csv"
HPO_TERMS_FILE = PROCESSED_DIR / "hpo_terms.csv"
HPO_EMBEDDINGS_FILE = PROCESSED_DIR / "hpo_embeddings.npy"

PUBMED_CACHE = PROCESSED_DIR / "pubmed_articles.csv"

FAISS_INDEX = VECTOR_DIR / "pubmed.index"
FAISS_METADATA = VECTOR_DIR / "pubmed_metadata.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_DISEASES = 5
TOP_ARTICLES = 8