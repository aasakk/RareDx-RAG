from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY


SYSTEM_PROMPT = """You are an evidence synthesis component in an educational rare-disease research prototype.

CRITICAL MANDATES:
1. You are NOT a physician and must NOT present the output as a clinical or confirmed diagnosis.
2. Use ONLY the evidence provided in the prompt. Do not extrapolate or invent genes, variants, symptoms, publications, PMIDs, statistics, or clinical facts.
3. If an evidence type (e.g., ClinVar variants or PubMed literature) is missing or empty, state explicitly: "None retrieved".
4. Never state or imply that retrieved ClinVar variants are carried by the patient. ClinVar records describe known pathogenic variants in candidate disease-associated genes.
5. For EACH candidate disease, you MUST format the output with the following exact Markdown sections:

### Candidate [Rank]: [Disease Name]
- **1. Candidate disease**: [Disease Name (Database ID)] — Experimental Compatibility Score: [Score]
- **2. Phenotype evidence**: [Concordant matched symptoms and degree of match (exact/hierarchy)]
- **3. Genetic evidence**: [Associated genes and retrieved ClinVar records, including the actual clinical significance classification when available. Never infer a classification. If unavailable, state 'Classification unavailable'.]
- **4. Literature evidence**: [Retrieved PubMed publications with explicit PMIDs, or 'None retrieved']
- **5. Conflicting or missing evidence**: [Unmatched symptoms, explicitly absent symptoms with penalties, or unannotated features]
- **6. Limitations**: [Specific limitations of the evidence for this candidate]

End your response with a clear medical and research disclaimer.
"""

def build_evidence_text(symptoms, candidates):
    blocks = [
        f"PATIENT-REPORTED PHENOTYPE QUERY:\n{symptoms.strip()}\n"
    ]

    for i, candidate in enumerate(candidates, start=1):
        scores = candidate.get("scores", {})
        final_sc = scores.get("final_score", candidate.get("score", 0.0))
        pheno_sc = scores.get("phenotype_score", candidate.get("score", 0.0))
        lit_sc = scores.get("literature_score", 0.0)
        gen_sc = scores.get("genetic_evidence_score", 0.0)

        block = [
            f"--- CANDIDATE {i} ---",
            f"Disease: {candidate['disease']} ({candidate.get('database_id', 'N/A')})",
            f"Experimental Score: {final_sc * 100:.1f}% "
            f"(Phenotype: {pheno_sc * 100:.1f}%, Literature: {lit_sc * 100:.1f}%, Genetic: {gen_sc * 100:.1f}%)",
            "Associated Genes: " + (", ".join(candidate.get("genes", [])) if candidate.get("genes") else "None mapped"),
            "Phenotype Matching Details:\n  " + "\n  ".join(candidate.get("matched_phenotypes", [])[:8]),
        ]

        articles = candidate.get("articles", [])
        if articles:
            art_lines = ["Retrieved PubMed Evidence:"]
            for art in articles[:3]:
                title = art.get('title', '').strip()
                pmid = art.get('pmid', '')
                year = art.get('year', '')
                art_lines.append(f"  - PMID {pmid} ({year}): {title[:120]}")
            block.append("\n".join(art_lines))
        else:
            block.append("Retrieved PubMed Evidence: None retrieved")

        variants = candidate.get("clinvar", [])
        if variants:
            var_lines = ["Retrieved ClinVar Genetic Evidence:"]
            for var in variants[:3]:
                title = var.get('title', var.get('variation', 'Pathogenic variant')).strip()
                gene = var.get('query_gene', '')
                classification = var.get("clinical_significance", {})
                if isinstance(classification, dict):
                    classification = (
                        classification.get("description")
                        or classification.get("clinical_significance")
                        or classification.get("review_status")
                        or "Classification unavailable"
                    )
                elif not classification:
                    classification = "Classification unavailable"

                var_lines.append(
                    f"  - {title[:120]} (Gene: {gene}; "
                    f"Clinical significance: {classification})"
                )
            block.append("\n".join(var_lines))
        else:
            block.append("Retrieved ClinVar Genetic Evidence: None retrieved")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def synthesize(symptoms, candidates):
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() in ["", "your-key-here", "your_gemini_api_key"]:
        return (
            "⚠️ **Gemini synthesis is disabled**: GEMINI_API_KEY is not configured in `.env`. "
            "Please provide a valid Gemini API key to generate natural language differential summaries."
        )

    evidence = build_evidence_text(symptoms, candidates)

    prompt = f"""EVIDENCE PAYLOAD:
{evidence}

Produce a complete, structured evidence-grounded differential summary for ALL candidates using the required section headers.
Ensure you complete the response without cutting off.
Conclude with a mandatory research prototype disclaimer.
"""

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    try:
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
    except Exception as init_exc:
        return f"Gemini client initialization failed: {init_exc}"

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )

            if not response or not response.text:
                return "⚠️ Gemini returned an empty response. Candidate ranking and evidence remains available above."

            return response.text.strip()

        except Exception as exc:
            last_error = exc
            continue

    return (
        "⚠️ **Gemini synthesis failed**: Unable to complete LLM synthesis.\n\n"
        f"**Error Details:** {last_error}\n\n"
        "The ranked candidate diseases, phenotype compatibility breakdown, ClinVar genetic variants, "
        "and PubMed literature evidence remain fully inspectable above."
    )