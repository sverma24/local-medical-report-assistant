from __future__ import annotations

import argparse
from pathlib import Path

# Focused subset for common panels and markers seen in routine reports.
KEEP_SLUGS = {
    "complete-blood-count-cbc",
    "basic-metabolic-panel-bmp",
    "comprehensive-metabolic-panel-cmp",
    "lipid-panel",
    "cholesterol-levels",
    "hba1c-hemoglobin-a1c",
    "blood-glucose-test",
    "glucose-in-urine-test",
    "urinalysis",
    "thyroid-stimulating-hormone-tsh-test",
    "t3-tests",
    "t4-thyroxine-test",
    "free-t4-test",
    "thyroid-antibodies",
    "vitamin-d-test",
    "vitamin-b-test",
    "vitamin-b12-test",
    "folate-test",
    "iron-tests",
    "ferritin-blood-test",
    "transferrin-test",
    "calcium-blood-test",
    "magnesium-blood-test",
    "phosphorus-blood-test",
    "electrolyte-panel",
    "sodium-blood-test",
    "potassium-blood-test",
    "chloride-blood-test",
    "bicarbonate-test",
    "kidney-function-tests",
    "creatinine-test",
    "blood-urea-nitrogen-bun-test",
    "estimated-glomerular-filtration-rate-egfr",
    "urine-albumin-creatinine-ratio",
    "liver-function-tests",
    "alanine-transaminase-alt-test",
    "aspartate-aminotransferase-ast-test",
    "alkaline-phosphatase",
    "bilirubin-blood-test",
    "gamma-glutamyl-transferase-ggt-test",
    "albumin-blood-test",
    "total-protein-and-albumin-globulin-a-g-ratio",
    "c-reactive-protein-crp-test",
    "sed-rate-erythrocyte-sedimentation-rate",
    "prothrombin-time-test-and-inr-ptinr",
    "partial-thromboplastin-time-ptt-test",
    "d-dimer-test",
    "lactate-dehydrogenase-ldh-test",
    "creatine-kinase",
    "troponin-test",
    "fibrinogen-blood-test",
    "white-blood-cell-wbc-in-stool",
    "reticulocyte-count",
    "mean-corpuscular-volume-mcv",
    "mean-corpuscular-hemoglobin-mch",
    "mean-corpuscular-hemoglobin-concentration-mchc",
    "red-cell-distribution-width-rdw",
    "platelet-tests",
    "hemoglobin-test",
    "hematocrit-test",
    "rbc-count",
    "wbc-white-blood-cell-count",
    "neutrophils",
    "lymphocytes",
    "monocytes",
    "eosinophil-count",
    "basophil-count",
    "uric-acid-test",
    "amylase-test",
    "lipase-tests",
    "lactic-acid-test",
    "osmolality-tests",
}


def prune_knowledge_dir(knowledge_dir: Path) -> tuple[int, int]:
    deleted = 0
    kept = 0
    for md_file in knowledge_dir.glob("*.md"):
        slug = md_file.stem
        if slug in KEEP_SLUGS:
            kept += 1
            continue
        md_file.unlink(missing_ok=True)
        deleted += 1
    return kept, deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete non-curated MedlinePlus markdown files in-place."
    )
    parser.add_argument(
        "--knowledge-dir",
        default="data/knowledge/medlineplus",
        help="Directory containing downloaded MedlinePlus markdown files.",
    )
    args = parser.parse_args()

    knowledge_dir = Path(args.knowledge_dir).resolve()
    kept, deleted = prune_knowledge_dir(knowledge_dir)
    print(f"knowledge_dir={knowledge_dir}")
    print(f"kept_files={kept}")
    print(f"deleted_files={deleted}")


if __name__ == "__main__":
    main()

