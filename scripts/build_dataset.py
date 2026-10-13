import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from src.config import (
    HPOA_FILE,
    GENES_FILE,
    DISEASE_DATASET,
)

from src.diseases import TARGET_DISEASES, MANUAL_GENE_MAPPING

from src.hpo import (
    download_hpo_data,
    load_hpo_terms,
)


# ---------------------------------------------------------
# LOAD HPO DISEASE → PHENOTYPE ANNOTATIONS
# ---------------------------------------------------------

def load_hpoa():
    print("[*] Loading phenotype.hpoa...")

    df = pd.read_csv(
        HPOA_FILE,
        sep="\t",
        comment="#",
        dtype=str,
    ).fillna("")

    print(f"[✓] Loaded {len(df):,} annotations")
    print("[*] Columns:", list(df.columns))

    return df


# ---------------------------------------------------------
# FIND OUR 18 TARGET DISEASES
# ---------------------------------------------------------

def find_disease_annotations(df):
    if "disease_name" not in df.columns:
        raise ValueError(
            "Could not find 'disease_name' in phenotype.hpoa.\n"
            f"Available columns: {list(df.columns)}"
        )

    output = []

    disease_names = df["disease_name"].str.lower()

    print("\n" + "=" * 70)
    print("SEARCHING FOR TARGET DISEASES")
    print("=" * 70)

    for display_name, aliases in TARGET_DISEASES.items():

        all_matches = []

        for alias in aliases:
            mask = disease_names.str.contains(
                alias.lower(),
                regex=False,
                na=False,
            )

            matches = df[mask]

            if not matches.empty:
                all_matches.append(matches)

        if not all_matches:
            print(f"[✗] NOT FOUND: {display_name}")
            continue

        disease_matches = pd.concat(
            all_matches,
            ignore_index=True,
        ).drop_duplicates()

        # One alias can sometimes match several OMIM/ORPHA records.
        # Determine the primary (representative) database_id:
        # prefer an exact name match, otherwise pick the richest annotation set.
        candidates = (
            disease_matches
            .groupby(
                ["database_id", "disease_name"]
            )["hpo_id"]
            .nunique()
            .reset_index(name="phenotype_count")
            .sort_values(
                "phenotype_count",
                ascending=False,
            )
        )

        exact_matches = candidates[
            candidates["disease_name"].str.lower().str.strip()
            == display_name.lower().strip()
        ]

        if not exact_matches.empty:
            primary = exact_matches.iloc[0]
        else:
            primary = candidates.iloc[0]

        primary_id = primary["database_id"]

        # Identify secondary records eligible for merging.
        # A secondary is eligible only if its disease_name is an exact or very
        # near match to display_name (handles OMIM parenthetical forms like
        # "Epileptic encephalopathy ... (Dravet syndrome)").
        # Records that merely contain display_name as a substring but have
        # additional qualifier words (e.g. "type I", "type IV", "distal") are
        # excluded to prevent explosion of phenotype sets for broad disease
        # categories like "Spinal muscular atrophy".
        dn_lower = display_name.lower().strip()
        merge_ids = [primary_id]

        for _, cand_row in candidates.iterrows():
            db_id = cand_row["database_id"]
            if db_id == primary_id:
                continue
            cand_name = cand_row["disease_name"].lower().strip()
            # Exact match (already in exact_matches) - include
            if cand_name == dn_lower:
                merge_ids.append(db_id)
                continue
            # Near-exact: cand_name ends with "(display_name)" parenthetical
            import re as _re
            if _re.search(r'\(' + _re.escape(dn_lower) + r'\)', cand_name):
                merge_ids.append(db_id)
                continue
            # Near-exact: cand_name IS display_name with only a comma-qualifier
            # e.g. "Dravet syndrome, familial" but NOT "Spinal muscular atrophy, type I"
            stripped = cand_name
            if stripped.startswith(dn_lower + ",") or stripped.startswith(dn_lower + " -"):
                # Allow only if there's no subtype indicator
                suffix = stripped[len(dn_lower):].strip(", -").strip()
                if suffix and not _re.search(r'\b(type|variant|subtype|form|\d|distal|proximal|x-linked|autosomal)\b', suffix):
                    merge_ids.append(db_id)

        # Merge eligible annotations (deduplicated per hpo_id).
        # Process primary record first so its frequency data takes precedence.
        merged_rows = []
        seen_hpo_ids = {}

        for db_id in merge_ids:
            subset = disease_matches[disease_matches["database_id"] == db_id]
            for _, row in subset.iterrows():
                h_id = row["hpo_id"]
                if h_id not in seen_hpo_ids:
                    seen_hpo_ids[h_id] = True
                    merged_rows.append(row)

        chosen = pd.DataFrame(merged_rows).copy()
        chosen["database_id"] = primary_id
        chosen["display_name"] = display_name

        merged_count = len(chosen["hpo_id"].unique()) if not chosen.empty else 0
        merge_ids_str = ", ".join(merge_ids[:4])
        print(
            f"[✓] {display_name}\n"
            f"    → primary: {primary_id} ({primary['disease_name']})\n"
            f"    → merged from: {merge_ids_str}\n"
            f"    → {merged_count} unique phenotypes after merge"
        )

        output.append(chosen)



    if not output:
        raise RuntimeError(
            "None of the target diseases were found."
        )

    result = pd.concat(
        output,
        ignore_index=True,
    )

    print(
        f"\n[✓] Successfully mapped "
        f"{result['display_name'].nunique()} target diseases."
    )

    return result


# ---------------------------------------------------------
# LOAD DISEASE → GENE MAPPING
# ---------------------------------------------------------

def load_gene_mapping():
    if not GENES_FILE.exists():
        print(
            "[!] genes_to_disease.txt wasn't downloaded. "
            "Continuing without gene mappings."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            GENES_FILE,
            sep="\t",
            dtype=str,
        ).fillna("")

        print(
            f"\n[✓] Loaded {len(df):,} disease-gene mappings"
        )

        print(
            "[*] Gene file columns:",
            list(df.columns),
        )

        return df

    except Exception as exc:
        print(
            "[!] Could not parse genes_to_disease.txt:",
            exc,
        )

        return pd.DataFrame()


# ---------------------------------------------------------
# ATTACH GENES
# ---------------------------------------------------------

def attach_genes(df, gene_df):
    df = df.copy()
    df["gene"] = ""

    if gene_df.empty:
        return df

    possible_disease_columns = [
        "disease_id",
        "database_id",
        "diseaseId",
    ]

    possible_gene_columns = [
        "gene_symbol",
        "geneSymbol",
        "gene",
    ]

    disease_column = next(
        (
            column
            for column in possible_disease_columns
            if column in gene_df.columns
        ),
        None,
    )

    gene_column = next(
        (
            column
            for column in possible_gene_columns
            if column in gene_df.columns
        ),
        None,
    )

    if disease_column is None or gene_column is None:
        print(
            "[!] Gene mapping columns were not recognized."
        )

        print(
            "    Available columns:",
            list(gene_df.columns),
        )

        return df

    mapping = (
        gene_df
        .groupby(disease_column)[gene_column]
        .apply(
            lambda values: "|".join(
                sorted(
                    {
                        value.strip()
                        for value in values
                        if value.strip()
                    }
                )
            )
        )
        .to_dict()
    )

    df["gene"] = (
        df["database_id"]
        .map(mapping)
        .fillna("")
    )

    # Apply manual gene overrides
    for idx, row in df.iterrows():
        display_name = row["display_name"]
        if display_name in MANUAL_GENE_MAPPING:
            existing_genes = set(row["gene"].split("|")) if row["gene"] else set()
            existing_genes.update(MANUAL_GENE_MAPPING[display_name])
            df.at[idx, "gene"] = "|".join(sorted([g for g in existing_genes if g]))

    mapped_count = (
        df.loc[df["gene"] != "", "display_name"]
        .nunique()
    )

    print(
        f"[✓] Gene mappings found for "
        f"{mapped_count} target diseases"
    )

    return df


# ---------------------------------------------------------
# BUILD FINAL DATASET
# ---------------------------------------------------------

def build_dataset():
    print("\n" + "=" * 70)
    print("RAREDX-RAG DATASET BUILDER")
    print("=" * 70)

    # Download HPO source files.
    download_hpo_data()

    # Parse HPO ontology.
    print("\n[*] Loading HPO terminology...")

    hpo_terms = load_hpo_terms()

    print(
        f"[✓] Loaded {len(hpo_terms):,} HPO terms"
    )

    hpo_name_map = dict(
        zip(
            hpo_terms["hpo_id"],
            hpo_terms["name"],
        )
    )

    # Load disease annotations.
    hpoa = load_hpoa()

    # Restrict dataset to our selected diseases.
    selected = find_disease_annotations(
        hpoa
    )

    # Convert HP:xxxx identifiers into readable names.
    selected["phenotype"] = (
        selected["hpo_id"]
        .map(hpo_name_map)
        .fillna("")
    )

    # Convert onset HP:xxxx identifiers into readable names where available.
    if "onset" in selected.columns:
        selected["onset"] = selected["onset"].apply(
            lambda x: hpo_name_map.get(x, x) if x else ""
        )

    # Add genes.
    gene_df = load_gene_mapping()

    selected = attach_genes(
        selected,
        gene_df,
    )

    # Keep only useful columns.
    desired_columns = [
        "display_name",
        "database_id",
        "disease_name",
        "gene",
        "hpo_id",
        "phenotype",
        "frequency",
        "onset",
        "evidence",
        "reference",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in selected.columns
    ]

    selected = selected[
        available_columns
    ].drop_duplicates()

    selected = selected.sort_values(
        by=[
            "display_name",
            "hpo_id",
        ]
    )

    # Save.
    DISEASE_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        DISEASE_DATASET,
        index=False,
    )

    return selected


# ---------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------

def print_summary(df):
    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    summary = (
        df
        .groupby("display_name")
        .agg(
            phenotype_count=(
                "hpo_id",
                "nunique",
            ),
            genes=(
                "gene",
                lambda values: "|".join(
                    sorted(
                        {
                            gene
                            for value in values
                            for gene in value.split("|")
                            if gene
                        }
                    )
                ),
            ),
        )
        .reset_index()
        .sort_values(
            "display_name"
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n------------------------------")

    print(
        "Diseases:",
        df["display_name"].nunique(),
    )

    print(
        "Unique HPO phenotypes:",
        df["hpo_id"].nunique(),
    )

    print(
        "Total annotations:",
        len(df),
    )

    print(
        "\nDataset saved to:",
        DISEASE_DATASET,
    )

    print(
        "\n[✓] RareDx dataset build complete."
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    dataset = build_dataset()

    print_summary(dataset)