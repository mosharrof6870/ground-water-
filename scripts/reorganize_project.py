"""
Project Reorganization and Clean-up Script for North Bengal Groundwater AI Screening.
Creates target directory structure, generates PROJECT_FILE_INVENTORY.csv and PROJECT_MANIFEST.csv,
safely archives intermediate/obsolete artifacts, and preserves all working models and app dependencies.
"""
import os
import shutil
import glob
import json
import pandas as pd
from datetime import datetime

ROOT_DIR = "/home/mosharrof/personal Doc/water jounal"

# Define Target Directories
DIRS_TO_CREATE = [
    "data/raw",
    "data/processed",
    "data/metadata",
    "scripts/01_data_validation",
    "scripts/02_preprocessing",
    "scripts/03_hydrochemistry",
    "scripts/04_risk_indices",
    "scripts/05_pca",
    "scripts/06_ml",
    "scripts/07_interpretation",
    "scripts/08_uncertainty",
    "scripts/09_validation",
    "models/ni",
    "models/cd",
    "results/tables",
    "results/figures",
    "results/metrics",
    "results/predictions",
    "results/uncertainty",
    "results/interpretation",
    "reports/final",
    "reports/audit",
    "reports/generated",
    "paper/tables",
    "paper/figures",
    "paper/manuscript",
    "docs/methodology",
    "docs/model_card",
    "docs/reproducibility",
    "archive/intermediate",
    "archive/obsolete",
    "archive/duplicates",
    "archive/debug"
]

def classify_file(rel_path, filename, ext, size):
    """
    Classifies a file into categories A to U based on its path, name, and type.
    """
    path_lower = rel_path.lower()
    fn_lower = filename.lower()
    
    # Hidden / git
    if rel_path.startswith(".") or "/." in rel_path:
        return "OBSOLETE", "Hidden system file", "UTILITY", "NO_DEPENDENCY", "UNIQUE", "ARCHIVE", "move_to_archive"

    # Log files
    if ext == ".log" or fn_lower == "streamlit.log":
        return "DEBUG_OUTPUT", "Runtime debug log", "DEBUG", "TEMPORARY", "TEMPORARY", "ARCHIVE", "move_to_archive_debug"
        
    # Lock files
    if fn_lower.startswith(".~lock"):
        return "DEBUG_OUTPUT", "Office temp lock file", "DEBUG", "TEMPORARY", "TEMPORARY", "ARCHIVE", "move_to_archive_debug"

    # Zip archives
    if ext == ".zip":
        return "OBSOLETE", "Compressed stage archive", "ARCHIVE", "HISTORICAL", "DUPLICATE_ZIP", "ARCHIVE", "move_to_archive_obsolete"

    # Application Files
    if rel_path.startswith("app/"):
        if "models/" in rel_path:
            if ext == ".joblib":
                return "FINAL_MODEL", "Trained ML Model Pipeline", "DEPLOYMENT", "CRITICAL_APP_DEP", "UNIQUE", "KEEP_ACTIVE", "preserve"
            elif ext == ".json":
                return "MODEL_ARTIFACT", "Training domain bounds JSON", "DEPLOYMENT", "CRITICAL_APP_DEP", "UNIQUE", "KEEP_ACTIVE", "preserve"
            elif "conformal" in fn_lower:
                return "UNCERTAINTY_ARTIFACT", "Conformal prediction calibration CSV", "DEPLOYMENT", "CRITICAL_APP_DEP", "UNIQUE", "KEEP_ACTIVE", "preserve"
        return "APPLICATION_SOURCE", "Streamlit UI/Service Source Code", "DEPLOYMENT", "CRITICAL_APP_DEP", "UNIQUE", "KEEP_ACTIVE", "preserve"

    # Raw Data
    if fn_lower == "groundwater quality data_northbengal.xlsx" or rel_path.startswith("data/raw"):
        return "FINAL_SOURCE_DATA", "Original North Bengal Groundwater Field Dataset", "DATA_INGESTION", "CRITICAL_SOURCE", "UNIQUE", "KEEP_ACTIVE", "preserve"

    # Processed Data
    if rel_path.startswith("data/processed") or fn_lower == "phase4_features_track1.csv":
        return "PROCESSED_DATA", "Cleaned Hydrochemical Feature Set", "PREPROCESSING", "CRITICAL_ML_DEP", "UNIQUE", "KEEP_ACTIVE", "preserve"

    # Final Results
    if rel_path.startswith("water_final_results/") or rel_path.startswith("results/"):
        if "/tables/" in path_lower or "table_" in fn_lower:
            return "FINAL_TABLE", "Publication-ready research table", "PUBLICATION", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"
        elif "/figures/" in path_lower or "figure_" in fn_lower or ext == ".png":
            return "FINAL_FIGURE", "Publication research figure", "PUBLICATION", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"
        elif "/uncertainty/" in path_lower or "conformal" in fn_lower:
            return "UNCERTAINTY_ARTIFACT", "Conformal uncertainty evaluation artifact", "UNCERTAINTY", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"
        elif "/predictions/" in path_lower or "oof_" in fn_lower:
            return "VALIDATION_ARTIFACT", "Out-of-fold predictions matrix", "VALIDATION", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"
        elif "/interpretation/" in path_lower or "shap" in fn_lower:
            return "SHAP_ARTIFACT", "SHAP interpretability output", "INTERPRETATION", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"
        elif "/metrics/" in path_lower:
            return "VALIDATION_ARTIFACT", "Model performance summary metrics", "VALIDATION", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"
        elif "/reports/" in path_lower:
            return "FINAL_REPORT", "Research summary report", "DOCUMENTATION", "RESEARCH_RESULT", "UNIQUE", "KEEP_ACTIVE", "preserve"

    # Reports
    if fn_lower == "scientific_audit_report.md" or rel_path.startswith("reports/"):
        return "FINAL_REPORT", "Formal Scientific Audit Report", "DOCUMENTATION", "CRITICAL_AUDIT", "UNIQUE", "KEEP_ACTIVE", "preserve"

    # Scripts
    if rel_path.startswith("scripts/"):
        return "RESEARCH_SCRIPT", "Pipeline or analysis research script", "PIPELINE", "CRITICAL_REPRO", "UNIQUE", "KEEP_ACTIVE", "preserve"

    # Root level Stage outputs & CSVs
    if fn_lower.startswith("stage") or fn_lower.startswith("phase"):
        if ext == ".md":
            return "FINAL_REPORT", "Stage Progress / Reconciliation Report", "DOCUMENTATION", "HISTORICAL_REPRO", "UNIQUE", "ARCHIVE", "move_to_archive_intermediate"
        elif ext == ".csv":
            return "INTERMEDIATE_ARTIFACT", "Stage intermediate tabular artifact", "INTERMEDIATE", "HISTORICAL_REPRO", "INTERMEDIATE", "ARCHIVE", "move_to_archive_intermediate"
        elif ext == ".json":
            return "INTERMEDIATE_ARTIFACT", "Stage configuration JSON", "INTERMEDIATE", "HISTORICAL_REPRO", "INTERMEDIATE", "ARCHIVE", "move_to_archive_intermediate"

    # Generic documentation
    if ext == ".md" or fn_lower in ["readme.md", "changelog.md", "license"]:
        return "DOCUMENTATION", "Project documentation / guide", "DOCUMENTATION", "REFERENCE", "UNIQUE", "KEEP_ACTIVE", "preserve"

    if ext == ".py":
        return "RESEARCH_SCRIPT", "Python research script", "PIPELINE", "CRITICAL_REPRO", "UNIQUE", "KEEP_ACTIVE", "preserve"

    return "UNKNOWN", "Uncategorized file", "UNKNOWN", "UNCHECKED", "UNKNOWN", "ARCHIVE", "move_to_archive_intermediate"


def main():
    print("=== STARTING GROUNDWATER RESEARCH CLEANUP & ORGANIZATION ===")
    
    # 1. Create directory structure
    for d in DIRS_TO_CREATE:
        full_d = os.path.join(ROOT_DIR, d)
        os.makedirs(full_d, exist_ok=True)
    print("[+] Verified target directory structure.")

    # 2. Copy raw data to data/raw/
    raw_src = os.path.join(ROOT_DIR, "Groundwater quality data_Northbengal.xlsx")
    raw_dst = os.path.join(ROOT_DIR, "data/raw/Groundwater quality data_Northbengal.xlsx")
    if os.path.exists(raw_src) and not os.path.exists(raw_dst):
        shutil.copy2(raw_src, raw_dst)
        print("[+] Copied primary raw Excel dataset to data/raw/")

    # 3. Inventory all files
    inventory = []
    manifest_rows = []

    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # Skip .git or .agents internal caches if desired, but index workspace files
        if "/.git" in dirpath or "/.venv" in dirpath or "/.streamlit" in dirpath:
            continue
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            ext = os.path.splitext(f)[1]
            size = os.path.getsize(full_path)
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            
            cat, purpose, stage, dep, dup, final_stat, action = classify_file(rel_path, f, ext, size)
            
            inventory.append({
                "path": rel_path,
                "filename": f,
                "extension": ext,
                "size_bytes": size,
                "modified_time": mtime,
                "category": cat,
                "purpose": purpose,
                "research_stage": stage,
                "dependency_status": dep,
                "duplicate_status": dup,
                "final_status": final_stat,
                "recommended_action": action
            })

            if final_stat == "KEEP_ACTIVE":
                manifest_rows.append({
                    "path": rel_path,
                    "filename": f,
                    "category": cat,
                    "size_bytes": size,
                    "modified_time": mtime
                })

    df_inv = pd.DataFrame(inventory)
    inv_path = os.path.join(ROOT_DIR, "PROJECT_FILE_INVENTORY.csv")
    df_inv.to_csv(inv_path, index=False)
    print(f"[+] Saved PROJECT_FILE_INVENTORY.csv ({len(df_inv)} files inventoried).")

    df_man = pd.DataFrame(manifest_rows)
    man_path = os.path.join(ROOT_DIR, "PROJECT_MANIFEST.csv")
    df_man.to_csv(man_path, index=False)
    print(f"[+] Saved PROJECT_MANIFEST.csv ({len(df_man)} active core files).")

    # 4. Safely move obsolete root-level stage CSVs/zips/logs into archive/
    archive_count = 0
    for item in inventory:
        action = item["recommended_action"]
        rel_path = item["path"]
        src_path = os.path.join(ROOT_DIR, rel_path)
        
        if not os.path.exists(src_path):
            continue

        # Never touch app/, data/, scripts/, models/, water_final_results/, scientific_audit_report.md directly
        if rel_path.startswith("app/") or rel_path.startswith("data/") or rel_path.startswith("models/") or rel_path == "scientific_audit_report.md" or rel_path in ["PROJECT_FILE_INVENTORY.csv", "PROJECT_MANIFEST.csv", "README.md"]:
            continue

        if action == "move_to_archive_debug":
            target_dir = os.path.join(ROOT_DIR, "archive/debug")
            shutil.move(src_path, os.path.join(target_dir, os.path.basename(rel_path)))
            archive_count += 1
        elif action == "move_to_archive_obsolete" or rel_path.endswith(".zip"):
            target_dir = os.path.join(ROOT_DIR, "archive/obsolete")
            shutil.move(src_path, os.path.join(target_dir, os.path.basename(rel_path)))
            archive_count += 1
        elif action == "move_to_archive_intermediate" and (rel_path.startswith("STAGE_") or rel_path.startswith("stage")):
            target_dir = os.path.join(ROOT_DIR, "archive/intermediate")
            shutil.move(src_path, os.path.join(target_dir, os.path.basename(rel_path)))
            archive_count += 1

    print(f"[+] Moved {archive_count} intermediate/obsolete/debug files into archive/.")
    print("=== REORGANIZATION COMPLETE ===")

if __name__ == "__main__":
    main()
