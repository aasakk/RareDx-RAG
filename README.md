# 🧬 RareDx-RAG

### Evidence-Grounded Rare Disease Diagnostic Support Using Hybrid Retrieval-Augmented Generation

RareDx-RAG is a research-oriented rare disease diagnostic support system that converts patient-reported phenotypes into a ranked list of candidate rare genetic diseases and enriches those candidates with phenotype, genetic, and biomedical literature evidence.

The system combines **Human Phenotype Ontology (HPO)** annotations, **ClinVar** genetic evidence, **PubMed** literature retrieval, semantic embeddings, **FAISS-based Retrieval-Augmented Generation (RAG)**, and **Gemini** to produce evidence-grounded and explainable candidate disease summaries.

> **Disclaimer:** RareDx-RAG is an educational and research prototype. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## 🎯 Problem Statement

Rare disease diagnosis is difficult because many disorders:

- share overlapping clinical phenotypes,
- have highly variable manifestations,
- involve uncommon genetic variants,
- are encountered infrequently in routine clinical practice.

Patients may therefore undergo a prolonged **diagnostic odyssey** before receiving an appropriate diagnosis.

A symptom such as developmental delay, seizures, muscle weakness, or speech impairment may occur across many genetic disorders. Simple keyword matching is therefore insufficient.

RareDx-RAG explores whether structured phenotype matching can be combined with genetic and literature retrieval to provide a more transparent candidate-ranking system.

---

## 💡 Core Idea

Given a set of patient phenotypes such as:

```text
seizures
developmental delay
speech impairment
```

RareDx-RAG:

1. Converts free-text symptoms into standardized **HPO terms**.
2. Compares them against known disease-phenotype annotations.
3. Ranks candidate rare diseases using phenotype compatibility.
4. Retrieves relevant biomedical literature from **PubMed**.
5. Retrieves known pathogenic genetic evidence from **ClinVar**.
6. Uses **FAISS-based semantic retrieval** to identify relevant literature.
7. Provides the retrieved evidence to **Gemini**.
8. Produces an evidence-grounded differential summary with citations.

---

# 🏗️ System Pipeline

```text
                  Patient Phenotypes
                         │
                         ▼
              ┌─────────────────────┐
              │ Phenotype Parsing   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ HPO Normalization   │
              │ Free text → HP IDs  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Phenotype Matching  │
              │                     │
              │ HPO disease         │
              │ annotations         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Disease Ranking     │
              └──────────┬──────────┘
                         │
                   Top Candidates
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │ ClinVar         │     │ PubMed          │
    │                 │     │                 │
    │ Gene / variant  │     │ Literature      │
    │ evidence        │     │ retrieval       │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             │                       ▼
             │              ┌─────────────────┐
             │              │ Embeddings      │
             │              │ + FAISS         │
             │              │ RAG Retrieval   │
             │              └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Evidence Fusion     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Gemini Synthesis    │
              └──────────┬──────────┘
                         │
                         ▼
              Ranked Candidate Diseases
                         +
                    Supporting Genes
                         +
                   ClinVar Evidence
                         +
                    PubMed Evidence
                         +
                       Citations
```

---

# 🧠 Hybrid Retrieval Architecture

RareDx-RAG deliberately separates **structured biomedical retrieval** from **unstructured literature retrieval**.

## Structured Retrieval

Structured sources are queried and processed directly:

```text
HPO
 ↓
Disease ↔ Phenotype

Disease
 ↓
Gene

Gene
 ↓
ClinVar variants
```

These sources are not unnecessarily converted into embeddings.

## Unstructured Retrieval

PubMed abstracts contain free-form biomedical text and are therefore handled through RAG:

```text
PubMed abstracts
       ↓
SentenceTransformer
       ↓
Vector embeddings
       ↓
FAISS vector index
       ↓
Semantic retrieval
       ↓
Relevant literature evidence
       ↓
Gemini
```

This creates a **hybrid retrieval system** rather than treating every biomedical database as unstructured text.

---

# 🧬 Current Disease Scope

The current prototype focuses on **18 rare genetic diseases** selected to intentionally contain overlapping neurological, developmental, metabolic, and neuromuscular phenotypes.

| Disease | Major Associated Gene(s) |
|---|---|
| Rett syndrome | MECP2 |
| Dravet syndrome | SCN1A and related genes |
| Angelman syndrome | UBE3A |
| Pitt-Hopkins syndrome | TCF4 |
| CDKL5 deficiency disorder | CDKL5 |
| Phelan-McDermid syndrome | SHANK3 |
| Kleefstra syndrome | EHMT1 |
| SLC6A1-related disorder | SLC6A1 |
| GLUT1 deficiency syndrome | SLC2A1 |
| Phenylketonuria | PAH |
| Wilson disease | ATP7B |
| Friedreich ataxia | FXN |
| Pompe disease | GAA |
| Spinal muscular atrophy | SMN1 |
| Duchenne muscular dystrophy | DMD |
| Fabry disease | GLA |
| Gaucher disease | GBA1 |
| Niemann-Pick disease type C | NPC1 / NPC2 |

The restricted disease set allows the complete retrieval and ranking pipeline to be evaluated without attempting to model thousands of rare diseases during the prototype stage.

---

# 📊 Dataset

The dataset is built automatically from HPO resources rather than manually defining phenotype lists.

The current dataset build contains approximately:

```text
18 target diseases
~1,000 disease-phenotype annotations
~650 unique HPO phenotypes
~20,000 HPO terminology entries
```

Exact counts may change as the upstream HPO dataset is updated.

The generated dataset contains fields including:

```text
display_name
database_id
disease_name
gene
hpo_id
phenotype
frequency
onset
evidence
reference
```

---

# 🔍 Phenotype Normalization

Users are not expected to know HPO identifiers.

For example:

```text
Input:
seizures
```

may be normalized to:

```text
Motor seizure
HP:0020219
```

RareDx-RAG uses sentence embeddings to compare patient-entered terminology with HPO names and synonyms.

This allows natural-language phenotype descriptions to be mapped to standardized biomedical terminology.

---

# 🧮 Disease Ranking

The current prototype combines exact HPO overlap and semantic phenotype similarity.

The phenotype matcher computes:

```text
Exact HPO compatibility
        +
Semantic phenotype similarity
        ↓
Phenotype score
```

Candidate diseases are then enriched using PubMed and ClinVar evidence.

The displayed score is an **experimental evidence-retrieval score**.

It is **not a probability that the patient has the disease**.

---

# 📚 PubMed RAG

For top candidate diseases, RareDx-RAG retrieves relevant PubMed articles using NCBI E-utilities.

Retrieved abstracts are transformed into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The vectors are indexed using **FAISS**.

For each candidate disease:

```text
Patient phenotype query
        ↓
Query embedding
        ↓
FAISS similarity search
        ↓
Top relevant abstracts
        ↓
Evidence supplied to Gemini
```

Each retrieved article retains its **PMID**, allowing literature evidence to remain traceable.

---

# 🧬 ClinVar Evidence

ClinVar is queried for pathogenic variants associated with genes linked to candidate diseases.

For example:

```text
Candidate:
Phelan-McDermid syndrome

Gene:
SHANK3

        ↓

ClinVar

        ↓

Known pathogenic SHANK3 variants
```

ClinVar evidence is used for **candidate enrichment**.

RareDx-RAG does **not** claim that a retrieved ClinVar variant is present in the patient because the current prototype does not receive patient genomic sequencing data.

---

# 🤖 Evidence-Grounded Generation

The final generation stage uses the **Gemini API**.

The LLM does not independently determine the initial disease ranking.

Instead, it receives evidence already retrieved by the system:

```text
Patient phenotypes
+
HPO matches
+
Ranked diseases
+
Associated genes
+
ClinVar evidence
+
Retrieved PubMed articles
        ↓
      Gemini
        ↓
Evidence-grounded explanation
```

The generation prompt explicitly instructs the model not to invent:

- genes,
- variants,
- symptoms,
- publications,
- PMIDs,
- clinical facts.

This separates **retrieval/ranking** from **natural-language evidence synthesis**.

---

# 🖥️ Interface

RareDx-RAG provides a Streamlit interface where users can:

- enter patient phenotypes,
- view normalized HPO terms,
- inspect ranked candidate diseases,
- inspect phenotype scores,
- view associated genes,
- inspect ClinVar evidence,
- read retrieved PubMed evidence,
- view retrieval similarity scores,
- receive an evidence-grounded summary.

---

# 🛠️ Technology Stack

### Core

```text
Python
Pandas
NumPy
Scikit-learn
```

### Biomedical Data

```text
Human Phenotype Ontology (HPO)
ClinVar
PubMed
NCBI E-utilities
```

### NLP / Retrieval

```text
Sentence Transformers
all-MiniLM-L6-v2
FAISS
Semantic Search
Retrieval-Augmented Generation
```

### Generation

```text
Google Gemini API
```

### Application

```text
Streamlit
```

---

# 📁 Project Structure

```text
raredx-rag/
│
├── app.py
├── requirements.txt
├── .env
│
├── data/
│   ├── raw/
│   │   ├── phenotype.hpoa
│   │   ├── hp.obo
│   │   └── genes_to_disease.txt
│   │
│   ├── processed/
│   │   ├── diseases.csv
│   │   ├── hpo_terms.csv
│   │   └── pubmed_articles.csv
│   │
│   └── vector_store/
│       ├── pubmed.index
│       └── pubmed_metadata.csv
│
├── scripts/
│   └── build_dataset.py
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── diseases.py
    ├── hpo.py
    ├── phenotype_matcher.py
    ├── pubmed.py
    ├── rag.py
    ├── clinvar.py
    ├── ranker.py
    └── synthesizer.py
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd raredx-rag
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment variables

Create:

```text
.env
```

Add:

```env
GEMINI_API_KEY=your_gemini_api_key
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=
```

`NCBI_API_KEY` is optional for the current prototype.

> Never commit `.env` or API keys to GitHub.

## 4. Build the biomedical dataset

```bash
python scripts/build_dataset.py
```

This downloads/processes the required HPO resources and generates the local disease-phenotype dataset.

## 5. Run the application

```bash
python -m streamlit run app.py
```

---

# 🚧 Implemented Architectural Updates

The current repository contains the finalized working pipeline.

The following architectural updates have been implemented to transform the system into a robust, phenotype-first diagnostic matcher.

## 1. Improved HPO Normalization

Current semantic normalization can occasionally map a broad phenotype to an unnecessarily specific HPO term.

Example:

```text
developmental delay

→ Mild global developmental delay
```

The normalization system prefers exact terminology and appropriate ontology-level matches before semantic fallback.

---

## 2. HPO Hierarchy-Aware Matching

HPO is hierarchical.

For example:

```text
Abnormality of movement
        ↓
Tremor
        ↓
Specific tremor phenotype
```

A patient phenotype and disease phenotype should therefore receive partial compatibility when they have a meaningful ancestor/descendant relationship even when their HPO IDs differ.

The matcher incorporates HPO ontology relationships rather than relying only on exact IDs and embedding similarity.

---

## 3. Severity-Aware Phenotypes

The interface allows each patient phenotype to include severity:

```text
Mild
Moderate
Severe
```

Example:

```text
Speech impairment     Severe
Tremor                Mild
Developmental delay   Severe
```

Severity will represent the relative importance of a phenotype in the patient's presentation.

A severe symptom will therefore contribute more strongly to phenotype compatibility than a minor symptom.

Severity alone will **not** be interpreted as evidence for a particular disease.

---

## 4. Phenotype Frequency

HPO disease annotations contain phenotype frequency information where available.

For example, a phenotype may be:

```text
Obligate
Very frequent
Frequent
Occasional
Very rare
```

Disease matching will incorporate this information.

Matching a phenotype that occurs in nearly all patients with a disease should generally provide stronger evidence than matching a phenotype that is only occasionally associated with the disease.

---

## 5. Age-of-Onset Compatibility

Patient phenotypes will optionally contain onset information.

Example:

```text
Seizures
Severity: Severe
Onset: Infancy
```

This can be compared against available HPO disease onset annotations.

The objective is to distinguish presentations such as:

```text
infantile-onset seizures

vs.

adult-onset seizures
```

when the underlying diseases have different typical onset profiles.

---

## 6. Negative Phenotypes

The system supports explicitly absent symptoms.

Example:

```text
Present:
✓ developmental delay
✓ seizures
✓ speech impairment

Absent:
✗ hypotonia
✗ tremor
```

Absence of a phenotype that is highly characteristic of a candidate disease can reduce its compatibility score.

This provides more information than considering only positive symptoms.

---

# 🧮 Phenotype Scoring

The phenotype score considers:

```text
Patient phenotype
        │
        ├── HPO similarity
        ├── ontology relationship
        ├── patient severity
        ├── disease phenotype frequency
        ├── onset compatibility
        └── positive / negative status
                ↓
        Phenotype compatibility
```

Conceptually:

```text
PhenotypeCompatibility =
    HPO Similarity
    × Patient Severity Weight
    × Disease Frequency Weight
    × Onset Compatibility
```

Negative phenotypes will contribute penalties where appropriate.

Phenotype compatibility will remain the dominant component of disease ranking.

PubMed and ClinVar evidence will primarily **support and explain** candidates rather than allowing literature volume to override poor phenotype compatibility.

---

# 🔎 Explainability

The system explicitly explains why each disease received its ranking.

Example:

```text
Candidate: Disease X

Strong Matches
✓ Severe speech impairment
  Disease frequency: Very frequent

✓ Developmental delay
  Disease frequency: Frequent

Partial Matches
~ Tremor
  Related phenotype found through HPO hierarchy

Conflicting Evidence
✗ Hypotonia absent in patient
  Hypotonia is very frequent in Disease X

Associated Gene
XYZ

Genetic Evidence
3 pathogenic ClinVar records retrieved

Literature Evidence
3 relevant PubMed articles retrieved
```

The goal is to make the ranking **inspectable rather than black-box**.

---

# 📈 Evaluation

The system is evaluated using known or constructed rare-disease phenotype cases.

For each test case:

```text
Known disease
      ↓
Patient phenotype set
      ↓
RareDx-RAG
      ↓
Ranked candidates
```

Performance will be reported using:

### Top-1 Accuracy

```text
Was the correct disease ranked #1?
```

### Top-3 Accuracy

```text
Was the correct disease among the first 3 candidates?
```

### Top-5 Accuracy

```text
Was the correct disease among the first 5 candidates?
```

Additional analysis will compare ranking performance before and after incorporating:

```text
HPO hierarchy
severity
frequency
onset
negative phenotypes
```

This provides an empirical way to evaluate whether the additional phenotype information improves candidate retrieval.

---

# 🗺️ Final Target Architecture

```text
                     PATIENT INPUT
                          │
          ┌───────────────┼────────────────┐
          │               │                │
      Phenotype        Severity          Onset
          │               │                │
          └───────────────┼────────────────┘
                          │
                   Present / Absent
                          │
                          ▼
                 HPO NORMALIZATION
                          │
                          ▼
               HPO HIERARCHY MATCHING
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ Phenotype Compatibility Engine  │
        │                                 │
        │ • semantic similarity           │
        │ • ontology similarity           │
        │ • severity weighting            │
        │ • phenotype frequency           │
        │ • onset compatibility           │
        │ • negative phenotype penalties  │
        └────────────────┬────────────────┘
                         │
                         ▼
                 DISEASE RANKING
                         │
                    Top Candidates
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
         CLINVAR                  PUBMED
             │                       │
       Genetic Evidence        Literature Search
             │                       │
             │                       ▼
             │                  Embeddings
             │                       │
             │                       ▼
             │                     FAISS
             │                       │
             │                       ▼
             │                RAG Retrieval
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                 EVIDENCE FUSION
                         │
                         ▼
                    GEMINI LLM
                         │
                         ▼
              EXPLAINABLE DIFFERENTIAL
                         │
              ┌──────────┼───────────┐
              │          │           │
          Phenotype    Genetic    Literature
          Evidence     Evidence    Evidence
                         │
                         ▼
                     EVALUATION
                  Top-1 / Top-3 / Top-5
```

---

# 🔬 Research Questions

RareDx-RAG explores the following questions:

1. Can natural-language patient symptoms be reliably normalized into standardized HPO phenotypes?

2. Does ontology-aware phenotype matching improve rare-disease candidate retrieval compared with simple semantic similarity?

3. Does incorporating symptom severity, frequency, onset, and negative phenotypes improve disease ranking?

4. Can structured genetic evidence and unstructured biomedical literature be combined through hybrid retrieval?

5. Can RAG provide useful supporting literature while keeping the underlying disease-ranking process transparent?

---

# ⚠️ Limitations

RareDx-RAG currently has several important limitations:

- The prototype covers only 18 selected diseases.
- Phenotype descriptions supplied by users may be incomplete or ambiguous.
- Semantic HPO normalization can produce incorrect mappings.
- HPO annotations vary in completeness between diseases.
- Frequency and onset metadata are not available for every phenotype.
- ClinVar evidence describes known disease-associated variants and does not establish that a patient carries those variants.
- PubMed retrieval depends on available indexed literature.
- LLM-generated explanations may contain errors despite grounding constraints.
- The ranking score is an experimental compatibility score, not a calibrated disease probability.
- The system has not undergone clinical validation.

---

# 🔐 Privacy

The prototype is intended for demonstration using **synthetic or de-identified phenotype descriptions**.

Personally identifiable medical information should not be entered into the application.

---

# 🧪 Example

### Input

```text
Seizures
Developmental delay
Speech impairment
```

### System processing

```text
Free-text phenotypes
        ↓
HPO normalization
        ↓
Candidate ranking
        ↓
PubMed RAG
        ↓
ClinVar enrichment
        ↓
Evidence synthesis
```

### Example output structure

```text
#1 Candidate Disease

Phenotype Compatibility: XX%

Associated Gene:
GENE

Matched Phenotypes:
• phenotype A
• phenotype B

Genetic Evidence:
• ClinVar pathogenic variant records

Literature Evidence:
• PMID XXXXXXXX
• PMID XXXXXXXX

Evidence Summary:
Evidence-grounded explanation generated from retrieved sources.
```

---

# 🌐 Data Sources

- **Human Phenotype Ontology (HPO)** — standardized human phenotype terminology and disease-phenotype annotations.
- **PubMed / NCBI** — biomedical literature retrieval.
- **ClinVar / NCBI** — relationships between genomic variants and human health.
- **NCBI E-utilities** — programmatic access to PubMed and ClinVar.

Users of the repository should review the respective databases' licenses, terms, and usage policies.

---

# 👥 Project Scope

RareDx-RAG was designed as a three-person academic side project with a deliberately restricted disease set.

The focus is not on building a production clinical diagnostic system.

Instead, the project demonstrates the integration of:

```text
Biomedical ontologies
+
Semantic NLP
+
Information retrieval
+
Vector search
+
Retrieval-Augmented Generation
+
Genetic databases
+
Explainable ranking
+
Quantitative evaluation
```

into a reproducible end-to-end biomedical information retrieval pipeline.

---

# 📌 Project Status

```text
[✓] HPO terminology ingestion
[✓] Automated disease-phenotype dataset builder
[✓] 18-disease prototype dataset
[✓] Free-text phenotype input
[✓] Embedding-based HPO normalization
[✓] Disease phenotype ranking
[✓] PubMed retrieval
[✓] SentenceTransformer embeddings
[✓] FAISS vector retrieval
[✓] ClinVar enrichment
[✓] Gemini integration
[✓] Streamlit interface

[✓] Improved HPO normalization
[✓] HPO hierarchy-aware matching
[✓] Severity-aware phenotype input
[✓] Phenotype-frequency weighting
[✓] Age-of-onset compatibility
[✓] Negative phenotype handling
[✓] Corrected phenotype-dominant ranking
[✓] Complete disease-gene mappings
[✓] Structured Gemini evidence synthesis
[✓] Candidate-level explainability
[✓] Top-1 / Top-3 / Top-5 evaluation
```

The project has successfully implemented these core phenotype matching and ranking components, completing the planned diagnostic pipeline.

---

# 📜 Disclaimer

RareDx-RAG is an experimental software project developed for educational and research purposes.

It does not provide medical advice, diagnosis, prognosis, or treatment recommendations.

Any clinical interpretation of symptoms, genetic variants, or biomedical literature must be performed by appropriately qualified healthcare professionals.