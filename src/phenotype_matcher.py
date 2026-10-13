import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DISEASE_DATASET,
    EMBEDDING_MODEL,
    HPO_EMBEDDINGS_FILE
)
from src.diseases import MANUAL_GENE_MAPPING
from src.hpo import (
    load_hpo_terms,
    split_symptoms,
    normalize_text
)

# Explicit mapping of HPO frequency ontology terms to representative weights
HPO_FREQ_MAP = {
    "hp:0040280": 1.0,   # Obligate (100%)
    "hp:0040281": 0.85,  # Very frequent (80-99%)
    "hp:0040282": 0.50,  # Frequent (30-79%)
    "hp:0040283": 0.15,  # Occasional (5-29%)
    "hp:0040284": 0.02,  # Very rare (1-4%)
    "hp:0040285": 0.0,   # Excluded (0%)
}

# Explicit patient-side severity weights
SEVERITY_WEIGHTS = {
    "mild": 0.5,
    "moderate": 1.0,
    "severe": 1.5,
    "profound": 1.5
}
DEFAULT_SEVERITY_WEIGHT = 1.0


def parse_frequency(freq_str):
    """
    Parses HPO frequency terms, percentages, and fractions.
    Returns:
        float in [0.0, 1.0] if frequency is known,
        None if frequency information is missing or unrecorded.
    """
    if not freq_str or pd.isna(freq_str):
        return None
    freq_str = str(freq_str).strip().lower()
    if not freq_str:
        return None

    # HPO frequency ontology terms
    for code, val in HPO_FREQ_MAP.items():
        if code in freq_str:
            return val

    # Percentages (e.g. '80%')
    pct_match = re.search(r'(\d+(?:\.\d+)?)%', freq_str)
    if pct_match:
        return min(1.0, max(0.0, float(pct_match.group(1)) / 100.0))

    # Fractions (e.g. '3/4')
    frac_match = re.search(r'(\d+)/(\d+)', freq_str)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        return min(1.0, max(0.0, num / den)) if den > 0 else None

    return None


class PhenotypeMatcher:

    def __init__(self):
        if not DISEASE_DATASET.exists():
            raise FileNotFoundError(
                "Disease dataset missing. Run "
                "`python scripts/build_dataset.py` first."
            )

        self.diseases = pd.read_csv(
            DISEASE_DATASET
        ).fillna("")

        self.hpo_terms = load_hpo_terms()

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # Build HPO DAG and fast lookup tables
        self.hpo_hierarchy = {}
        self.hpo_ancestors = {}
        self.hpo_by_id = {}
        self.name_to_id = {}
        self.synonym_to_ids = {}

        for _, row in self.hpo_terms.iterrows():
            h_id = row["hpo_id"]
            name = row["name"]
            parents = row["parents"].split("|") if row["parents"] else []
            self.hpo_hierarchy[h_id] = parents
            self.hpo_by_id[h_id] = row

            norm_name = normalize_text(name)
            if norm_name and norm_name not in self.name_to_id:
                self.name_to_id[norm_name] = h_id

            if row["synonyms"]:
                for syn in row["synonyms"].split("|"):
                    norm_syn = normalize_text(syn)
                    if norm_syn:
                        self.synonym_to_ids.setdefault(norm_syn, []).append(h_id)

        self._prepare_hpo_embeddings()

    def get_ancestors(self, hpo_id):
        """Recursively traverses the HPO DAG to find all ancestor terms."""
        if hpo_id in self.hpo_ancestors:
            return self.hpo_ancestors[hpo_id]
        ancestors = set()
        for parent in self.hpo_hierarchy.get(hpo_id, []):
            if parent:
                ancestors.add(parent)
                ancestors.update(self.get_ancestors(parent))
        self.hpo_ancestors[hpo_id] = ancestors
        return ancestors

    def _prepare_hpo_embeddings(self):
        """Loads precomputed HPO embeddings from disk if available, or computes and caches them."""
        expected_shape = (len(self.hpo_terms), 384)
        if HPO_EMBEDDINGS_FILE.exists():
            try:
                embeddings = np.load(HPO_EMBEDDINGS_FILE)
                if embeddings.shape == expected_shape:
                    self.hpo_embeddings = embeddings
                    return
            except Exception:
                pass

        texts = []
        for _, row in self.hpo_terms.iterrows():
            text = row["name"]
            if row["synonyms"]:
                text += ". " + row["synonyms"].replace("|", ". ")
            texts.append(text)

        self.hpo_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        try:
            HPO_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            np.save(HPO_EMBEDDINGS_FILE, self.hpo_embeddings)
        except Exception:
            pass

    def symptom_to_hpo(self, symptom, top_k=3, requested_severity=None):
        """
        Normalizes a clean symptom string to HPO.
        Prefers:
          1. Exact term-name match
          2. Exact synonym match (selecting generic ancestor when appropriate)
          3. Semantic similarity fallback
        """
        norm_symp = normalize_text(symptom)

        # 1. Exact Name Match
        if norm_symp in self.name_to_id:
            h_id = self.name_to_id[norm_symp]
            row = self.hpo_by_id[h_id]
            return [{
                "input_symptom": symptom,
                "hpo_id": h_id,
                "hpo_name": row["name"],
                "similarity": 1.0
            }]

        # 2. Exact Synonym Match
        if norm_symp in self.synonym_to_ids:
            candidate_ids = self.synonym_to_ids[norm_symp]
            selected_id = candidate_ids[0]

            # If multiple terms share this synonym, avoid choosing an unnecessarily specific
            # term with severity modifiers unless explicitly requested.
            if len(candidate_ids) > 1:
                # Check for an ancestor / general term
                for cand_id in candidate_ids:
                    cand_name = self.hpo_by_id[cand_id]["name"].lower()
                    has_sev_modifier = any(
                        m in cand_name for m in ["mild ", "moderate ", "severe ", "profound "]
                    )
                    if not requested_severity and not has_sev_modifier:
                        selected_id = cand_id
                        break
                    # If cand_id is an ancestor of others, prefer it
                    cand_ancestors = self.get_ancestors(cand_id)
                    if any(other in self.get_ancestors(cand_id) for other in candidate_ids if other != cand_id):
                        selected_id = cand_id
                        break

            row = self.hpo_by_id[selected_id]
            return [{
                "input_symptom": symptom,
                "hpo_id": selected_id,
                "hpo_name": row["name"],
                "similarity": 0.99
            }]

        # 3. Semantic Fallback
        embedding = self.model.encode(
            [symptom],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        scores = cosine_similarity(
            embedding,
            self.hpo_embeddings
        )[0]

        top_indices = np.argsort(scores)[::-1][:top_k * 3]

        results = []
        for index in top_indices:
            row = self.hpo_terms.iloc[index]
            score = float(scores[index])
            name_lower = row["name"].lower()

            # When the user did not specify a severity, avoid mapping to an overly specific
            # subtype with severity modifiers if a broader ancestor is also nearby in score
            if not requested_severity:
                is_modified = any(
                    m in name_lower for m in ["mild ", "moderate ", "severe ", "profound "]
                )
                if is_modified:
                    # Look for parent term
                    parents = self.hpo_hierarchy.get(row["hpo_id"], [])
                    found_parent = False
                    for p in parents:
                        if p in self.hpo_by_id:
                            p_row = self.hpo_by_id[p]
                            p_name = p_row["name"].lower()
                            if not any(m in p_name for m in ["mild ", "moderate ", "severe "]):
                                results.append({
                                    "input_symptom": symptom,
                                    "hpo_id": p_row["hpo_id"],
                                    "hpo_name": p_row["name"],
                                    "similarity": max(score - 0.02, 0.75)
                                })
                                found_parent = True
                                break
                    if found_parent:
                        continue

            results.append({
                "input_symptom": symptom,
                "hpo_id": row["hpo_id"],
                "hpo_name": row["name"],
                "similarity": score
            })

            if len(results) >= top_k:
                break

        return results[:top_k]

    def normalize_symptoms(self, symptom_text):
        """
        Parses free-text symptoms, extracting severity and negation,
        while preserving actual pathological terms like 'absent speech'.
        """
        symptoms = split_symptoms(symptom_text)
        results = []

        severity_pattern = re.compile(r'\b(mild|moderate|severe|profound)\b', re.IGNORECASE)

        for raw_symptom in symptoms:
            raw_symptom_norm = normalize_text(raw_symptom)
            if not raw_symptom_norm:
                continue

            # Check if raw_symptom itself is directly a known HPO term or synonym
            # (e.g. 'absent speech' -> HP:0001344, 'absent reflexes' -> HP:0001284)
            is_direct_term = (
                raw_symptom_norm in self.name_to_id
                or raw_symptom_norm in self.synonym_to_ids
            )

            is_negative = False
            clean_symptom = raw_symptom

            if not is_direct_term:
                # Check for explicit negation prefixes or suffixes
                neg_prefix = re.match(r'^(no|without|denies|lack of|free of)\s+(.*)', raw_symptom, re.IGNORECASE)
                neg_suffix = re.search(r'\s+(is absent|absent|not present|excluded)$', raw_symptom, re.IGNORECASE)

                if neg_prefix:
                    is_negative = True
                    clean_symptom = neg_prefix.group(2).strip()
                elif neg_suffix:
                    is_negative = True
                    clean_symptom = raw_symptom[:neg_suffix.start()].strip()

            # Extract severity
            sev_match = severity_pattern.search(clean_symptom)
            severity = sev_match.group(1).lower() if sev_match else None
            severity_weight = SEVERITY_WEIGHTS.get(severity, DEFAULT_SEVERITY_WEIGHT)

            # Clean residual symptom for HPO lookup
            if not is_direct_term:
                clean_symptom = severity_pattern.sub('', clean_symptom)
                clean_symptom = re.sub(r'\s+', ' ', clean_symptom).strip()

            if not clean_symptom:
                continue

            matches = self.symptom_to_hpo(
                clean_symptom,
                top_k=1,
                requested_severity=severity
            )

            if matches:
                match = matches[0]
                match["severity"] = severity
                match["severity_weight"] = severity_weight
                match["is_negative"] = is_negative
                match["original_symptom"] = raw_symptom
                results.append(match)

        return results

    def rank_diseases(self, symptom_text, patient_onset=None):
        """
        Ranks candidate rare diseases using:
          - Ontology hierarchy (exact vs descendant vs ancestor)
          - No double-counting against the same disease annotation
          - Explicit patient severity weights (mild 0.5, moderate 1.0, severe 1.5)
          - HPO phenotype frequency scaling
          - Explicitly absent phenotype penalties
          - Bounded age-of-onset compatibility
        """
        normalized = self.normalize_symptoms(symptom_text)

        if not normalized:
            return [], []

        valid_symptoms = [x for x in normalized if x["similarity"] >= 0.35]
        if not valid_symptoms:
            return [], normalized

        # Total patient severity weight for normalization
        total_patient_weight = sum(s["severity_weight"] for s in valid_symptoms)

        results = []

        for display_name, group in self.diseases.groupby("display_name"):
            disease_hpos = {}
            for _, row in group.iterrows():
                freq_val = parse_frequency(row.get("frequency", ""))
                disease_hpos[row["hpo_id"]] = {
                    "freq": freq_val,
                    "name": row["phenotype"] or row["hpo_id"]
                }

            score_sum = 0.0
            matched_details = []
            used_disease_hpos = set()

            # 1. Evaluate each patient symptom
            for p_symp in valid_symptoms:
                p_id = p_symp["hpo_id"]
                p_ancestors = self.get_ancestors(p_id)
                p_weight = p_symp["severity_weight"]
                p_sev_label = f"[{p_symp['severity']}]" if p_symp.get("severity") else ""

                best_match_score = 0.0
                best_match_type = "unrelated"
                best_d_id = None

                # Search through available disease annotations (preventing double-counting)
                for d_id, d_info in disease_hpos.items():
                    if d_id in used_disease_hpos:
                        continue

                    d_ancestors = self.get_ancestors(d_id)
                    m_score = 0.0
                    m_type = "unrelated"

                    if p_id == d_id:
                        m_score = 1.0
                        m_type = "exact"
                    elif d_id in p_ancestors:
                        m_score = 0.8  # patient term is descendant of disease term
                        m_type = "descendant"
                    elif p_id in d_ancestors:
                        m_score = 0.6  # patient term is ancestor of disease term
                        m_type = "ancestor"

                    if m_score > best_match_score:
                        best_match_score = m_score
                        best_match_type = m_type
                        best_d_id = d_id

                if best_match_score > 0 and best_d_id is not None:
                    used_disease_hpos.add(best_d_id)
                    d_freq = disease_hpos[best_d_id]["freq"]
                    d_name = disease_hpos[best_d_id]["name"]

                    if p_symp.get("is_negative"):
                        # Negative / explicitly absent symptom
                        if d_freq is not None:
                            if d_freq >= 0.7:
                                # Very frequently expected in disease but absent in patient -> penalize
                                penalty = -1.0 * d_freq * p_weight
                                score_sum += penalty
                                matched_details.append(
                                    f"! Explicitly absent: '{p_symp['original_symptom']}' "
                                    f"(frequent in {display_name}: {d_name} [{best_d_id}], freq {d_freq*100:.0f}%, penalty applied)"
                                )
                            elif d_freq > 0.0:
                                # Occasional/rare in disease -> smaller penalty
                                penalty = -0.2 * d_freq * p_weight
                                score_sum += penalty
                                matched_details.append(
                                    f"✓ Explicitly absent: '{p_symp['original_symptom']}' "
                                    f"(rare/occasional in {display_name}: {d_name} [{best_d_id}], minimal penalty)"
                                )
                            else:
                                matched_details.append(
                                    f"✓ Explicitly absent: '{p_symp['original_symptom']}' "
                                    f"(excluded in {display_name}: {d_name} [{best_d_id}])"
                                )
                        else:
                            # Frequency is missing/unrecorded: apply no unsupported penalty
                            matched_details.append(
                                f"~ Explicitly absent: '{p_symp['original_symptom']}' "
                                f"(frequency unrecorded in {display_name}: {d_name} [{best_d_id}], no penalty)"
                            )
                    else:
                        # Positive symptom match
                        freq_weight = (0.5 + 0.5 * d_freq) if d_freq is not None else 0.75
                        contribution = best_match_score * freq_weight * p_weight
                        score_sum += contribution

                        freq_desc = f"{d_freq*100:.0f}%" if d_freq is not None else "unknown freq"
                        matched_details.append(
                            f"✓ {p_symp['original_symptom']} {p_sev_label} "
                            f"({best_match_type} match to '{d_name}' [{best_d_id}], freq: {freq_desc}, weight: {p_weight})"
                        )
                else:
                    if p_symp.get("is_negative"):
                        matched_details.append(
                            f"✓ Explicitly absent: '{p_symp['original_symptom']}' "
                            f"(not associated with {display_name})"
                        )
                    else:
                        matched_details.append(
                            f"✗ Unmatched: '{p_symp['original_symptom']}' {p_sev_label}"
                        )

            # Raw phenotype compatibility (normalized by total patient weight)
            raw_phenotype_score = max(0.0, score_sum / max(total_patient_weight, 1.0))

            # 2. Onset compatibility check (bounded adjustment +/-0.05)
            onset_adjustment = 0.0
            if patient_onset and "onset" in group.columns:
                d_onsets = [
                    str(o).strip() for o in group["onset"].dropna().unique()
                    if str(o).strip()
                ]

                if d_onsets:
                    onset_emb = self.model.encode(
                        [patient_onset],
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    d_onset_emb = self.model.encode(
                        d_onsets,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    onset_sim = cosine_similarity(onset_emb, d_onset_emb).max()

                    if onset_sim >= 0.55:
                        onset_adjustment = 0.05
                        matched_details.append(
                            f"✓ Onset concordant: patient '{patient_onset}' matches disease onset "
                            f"({', '.join(d_onsets[:2])}) [+0.05]"
                        )
                    elif onset_sim < 0.20:
                        onset_adjustment = -0.05
                        matched_details.append(
                            f"✗ Onset discordant: patient '{patient_onset}' clashes with disease onset "
                            f"({', '.join(d_onsets[:2])}) [-0.05]"
                        )
                    else:
                        matched_details.append(
                            f"~ Onset: '{patient_onset}' partially aligned with {', '.join(d_onsets[:2])}"
                        )
                else:
                    matched_details.append(
                        "~ Onset: disease records contain no onset data (neutral, no penalty)"
                    )

            final_phenotype_score = max(0.0, min(1.0, raw_phenotype_score + onset_adjustment))

            # Extract associated genes
            genes = []
            if "gene" in group.columns:
                for val in group["gene"]:
                    genes.extend([x for x in str(val).split("|") if x.strip()])

            # Supplement from manual mapping if empty
            if not genes and display_name in MANUAL_GENE_MAPPING:
                genes.extend(MANUAL_GENE_MAPPING[display_name])

            results.append({
                "disease": display_name,
                "database_id": group.iloc[0]["database_id"],
                "genes": sorted(set(genes)),
                "score": final_phenotype_score,
                "exact_score": raw_phenotype_score,
                "semantic_score": 0.0,
                "matched_phenotypes": matched_details,
                "phenotype_count": len(disease_hpos)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results, normalized