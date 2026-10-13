import re
import requests
import pandas as pd

from src.config import (
    HPOA_URL,
    HPO_OBO_URL,
    GENES_TO_DISEASE_URL,
    HPOA_FILE,
    HPO_OBO_FILE,
    GENES_FILE,
    HPO_TERMS_FILE,
)


def download_file(url, destination):
    if destination.exists():
        return

    print(f"Downloading {destination.name}...")

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    destination.write_bytes(response.content)


def download_hpo_data():
    download_file(HPOA_URL, HPOA_FILE)
    download_file(HPO_OBO_URL, HPO_OBO_FILE)

    try:
        download_file(GENES_TO_DISEASE_URL, GENES_FILE)
    except Exception as exc:
        print("Gene mapping download failed:", exc)


def parse_hpo_obo():
    terms = []

    current = {}

    with open(HPO_OBO_FILE, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "[Term]":
                if current.get("id") and current.get("name"):
                    terms.append(current)

                current = {}

            elif line.startswith("id: HP:"):
                current["id"] = line.split("id: ", 1)[1]

            elif line.startswith("name: "):
                current["name"] = line.split("name: ", 1)[1]

            elif line.startswith("synonym: "):
                match = re.search(r'"(.+?)"', line)

                if match:
                    current.setdefault("synonyms", [])
                    current["synonyms"].append(match.group(1))

            elif line.startswith("is_a: HP:"):
                # e.g., is_a: HP:0001263 ! Global developmental delay
                parent_id = line.split("is_a: ", 1)[1].split(" ")[0]
                current.setdefault("parents", [])
                current["parents"].append(parent_id)

    if current.get("id") and current.get("name"):
        terms.append(current)

    rows = []

    for term in terms:
        rows.append({
            "hpo_id": term["id"],
            "name": term["name"],
            "synonyms": "|".join(term.get("synonyms", [])),
            "parents": "|".join(term.get("parents", []))
        })

    df = pd.DataFrame(rows)

    df.to_csv(HPO_TERMS_FILE, index=False)

    return df


def load_hpo_terms():
    if not HPO_TERMS_FILE.exists():
        download_hpo_data()
        return parse_hpo_obo()

    df = pd.read_csv(HPO_TERMS_FILE).fillna("")
    if "parents" not in df.columns:
        return parse_hpo_obo()
    return df


def normalize_text(text):
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_symptoms(text):
    symptoms = re.split(r"[,;\n]+", text)

    return [
        normalize_text(x)
        for x in symptoms
        if normalize_text(x)
    ]