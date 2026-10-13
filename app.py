import streamlit as st

from src.config import TOP_DISEASES
from src.phenotype_matcher import PhenotypeMatcher
from src.rag import PubMedRAG
from src.clinvar import get_clinvar_evidence
from src.ranker import calculate_final_score
from src.synthesizer import synthesize


st.set_page_config(
    page_title="RareDx-RAG",
    page_icon="🧬",
    layout="wide"
)


st.title("🧬 RareDx-RAG")

st.caption(
    "Evidence-Grounded Rare Disease Diagnostic Support "
    "Using Hybrid Retrieval-Augmented Generation"
)

st.warning(
    "Research prototype only. This application does not "
    "provide medical diagnoses and must not be used for "
    "clinical decision-making."
)


@st.cache_resource
def get_matcher():
    return PhenotypeMatcher()


@st.cache_resource
def get_rag():
    return PubMedRAG()


try:
    matcher = get_matcher()
    rag = get_rag()

except FileNotFoundError:
    st.error(
        "Dataset not found. Run:\n\n"
        "`python scripts/build_dataset.py`"
    )

    st.stop()


st.subheader("Patient phenotype input")

symptoms = st.text_area(
    "Enter symptoms / phenotypes (include 'no' or 'absent' for negative phenotypes)",
    placeholder=(
        "Example: seizures, developmental delay, "
        "speech impairment, no hearing loss"
    ),
    height=120
)

patient_onset = st.text_input(
    "Patient age of onset (optional)",
    placeholder="Example: infancy, childhood, adult, 2 years"
)


if st.button(
    "Analyze phenotypes",
    type="primary"
):

    if not symptoms.strip():
        st.error(
            "Enter at least one phenotype."
        )

        st.stop()

    with st.spinner(
        "Matching phenotypes..."
    ):
        ranked, normalized = (
            matcher.rank_diseases(
                symptoms,
                patient_onset
            )
        )

    candidates = ranked[
        :TOP_DISEASES
    ]

    if not candidates:
        st.error(
            "No candidate diseases were generated."
        )

        st.stop()

    st.subheader(
        "Phenotype normalization"
    )

    for item in normalized:
        sev_str = f" | Severity: **{item['severity'].capitalize()}** (wt: {item['severity_weight']})" if item.get('severity') else ""
        status_str = " | ❌ **Explicitly Absent**" if item.get('is_negative') else " | ✅ **Present**"
        orig_str = item.get('original_symptom', item['input_symptom'])
        st.write(
            f"• **{orig_str}** "
            f"→ {item['hpo_name']} (`{item['hpo_id']}`) "
            f"— match confidence {item['similarity']:.2f}{sev_str}{status_str}"
        )

    with st.spinner(
        "Retrieving biomedical literature..."
    ):
        rag.build_for_candidates(
            candidates
        )

    enriched = []

    for candidate in candidates:

        disease = candidate[
            "disease"
        ]

        genes = candidate.get(
            "genes",
            []
        )

        with st.spinner(
            f"Gathering evidence for {disease}..."
        ):
            articles = rag.retrieve(
                query=(
                    symptoms
                    + " "
                    + disease
                ),
                candidate_disease=disease,
                top_k=4
            )

            clinvar = (
                get_clinvar_evidence(
                    genes,
                    disease,
                    per_gene=2
                )
                if genes
                else []
            )

        scores = calculate_final_score(
            candidate["score"],
            articles,
            clinvar
        )

        enriched.append({
            **candidate,
            "articles": articles,
            "clinvar": clinvar,
            "scores": scores
        })

    enriched.sort(
        key=lambda x:
        x["scores"]["final_score"],
        reverse=True
    )

    st.divider()

    st.header(
        "Ranked candidate diseases"
    )

    for rank, candidate in enumerate(
        enriched,
        start=1
    ):

        score = (
            candidate["scores"][
                "final_score"
            ]
            * 100
        )

        with st.expander(
            f"#{rank} {candidate['disease']} "
            f"— {score:.1f}% evidence score",
            expanded=(rank == 1)
        ):

            col1, col2, col3 = (
                st.columns(3)
            )

            col1.metric(
                "Phenotype",
                f"{candidate['scores']['phenotype_score'] * 100:.1f}%"
            )

            col2.metric(
                "Literature",
                f"{candidate['scores']['literature_score'] * 100:.1f}%"
            )

            col3.metric(
                "Genetic evidence",
                f"{candidate['scores']['genetic_evidence_score'] * 100:.1f}%"
            )

            st.markdown(
                "### Genes"
            )

            if candidate["genes"]:
                st.write(
                    ", ".join(
                        candidate["genes"]
                    )
                )
            else:
                st.write(
                    "No gene mapping available "
                    "in the current dataset."
                )

            st.markdown(
                "### Matched phenotypes"
            )

            if candidate[
                "matched_phenotypes"
            ]:
                for phenotype in candidate[
                    "matched_phenotypes"
                ]:
                    st.write(
                        f"• {phenotype}"
                    )
            else:
                st.write(
                    "Match primarily based on "
                    "semantic phenotype similarity."
                )

            st.markdown(
                "### PubMed evidence"
            )

            if candidate["articles"]:
                for article in candidate[
                    "articles"
                ]:
                    st.markdown(
                        f"**{article['title']}**"
                    )

                    st.caption(
                        f"PMID: {article['pmid']} "
                        f"| {article['year']} "
                        f"| retrieval similarity: "
                        f"{article['similarity']:.2f}"
                    )

                    if article[
                        "abstract"
                    ]:
                        abstract = article[
                            "abstract"
                        ]

                        st.write(
                            abstract[:600]
                            + (
                                "..."
                                if len(
                                    abstract
                                ) > 600
                                else ""
                            )
                        )
            else:
                st.write(
                    "No PubMed abstract retrieved."
                )

            st.markdown(
                "### ClinVar evidence"
            )

            if candidate["clinvar"]:

                for variant in candidate[
                    "clinvar"
                ][:5]:

                    st.write(
                        f"• {variant.get('title', 'Variant')}"
                    )

                    st.caption(
                        "Associated query gene: "
                        + variant.get(
                            "query_gene",
                            ""
                        )
                    )

            else:
                st.write(
                    "No ClinVar evidence retrieved "
                    "for the mapped genes."
                )

            st.caption(
                "Note: ClinVar evidence reflects known pathogenic variants associated with the candidate disease's genes. "
                "It does not indicate or imply that the patient carries these variants."
            )

    st.divider()

    st.header(
        "Evidence synthesis"
    )

    with st.spinner(
        "Generating evidence-grounded summary..."
    ):
        summary = synthesize(
            symptoms,
            enriched
        )

    st.markdown(summary)

    st.info(
        "The ranking is an experimental evidence-retrieval "
        "score, not a disease probability or clinical diagnosis."
    )