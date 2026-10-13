import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX,
    FAISS_METADATA
)

from src.pubmed import get_disease_articles


class PubMedRAG:

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.index = None
        self.metadata = pd.DataFrame()

        self.load()

    def load(self):
        if (
            FAISS_INDEX.exists()
            and FAISS_METADATA.exists()
        ):
            self.index = faiss.read_index(
                str(FAISS_INDEX)
            )

            self.metadata = pd.read_csv(
                FAISS_METADATA
            ).fillna("")

    def build_for_candidates(self, candidates):
        rows = []

        seen_pmids = set()

        for candidate in candidates:
            disease = candidate["disease"]
            genes = candidate.get(
                "genes",
                []
            )

            try:
                articles = get_disease_articles(
                    disease,
                    genes,
                    max_results=8
                )
            except Exception as exc:
                print(
                    f"PubMed failed for {disease}:",
                    exc
                )
                continue

            for article in articles:
                if not article["abstract"]:
                    continue

                if article["pmid"] in seen_pmids:
                    continue

                seen_pmids.add(
                    article["pmid"]
                )

                text = (
                    f"Disease: {disease}. "
                    f"Title: {article['title']}. "
                    f"Abstract: {article['abstract']}"
                )

                rows.append({
                    **article,
                    "disease": disease,
                    "text": text
                })

        if not rows:
            self.index = None
            self.metadata = pd.DataFrame()
            return

        self.metadata = pd.DataFrame(
            rows
        )

        embeddings = self.model.encode(
            self.metadata["text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        faiss.write_index(
            self.index,
            str(FAISS_INDEX)
        )

        self.metadata.to_csv(
            FAISS_METADATA,
            index=False
        )

    def retrieve(
        self,
        query,
        candidate_disease=None,
        top_k=5
    ):
        if (
            self.index is None
            or self.metadata.empty
        ):
            return []

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        # Retrieve extra results so disease filtering
        # still leaves enough articles.
        search_k = min(
            max(top_k * 5, top_k),
            len(self.metadata)
        )

        scores, indices = self.index.search(
            query_vector,
            search_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):
            if index < 0:
                continue

            row = self.metadata.iloc[
                index
            ]

            if (
                candidate_disease
                and row["disease"]
                != candidate_disease
            ):
                continue

            results.append({
                "pmid": row["pmid"],
                "title": row["title"],
                "abstract": row["abstract"],
                "disease": row["disease"],
                "year": row["year"],
                "similarity": float(score)
            })

            if len(results) >= top_k:
                break

        return results