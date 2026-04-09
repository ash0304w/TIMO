import csv
import json
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dump_candidate_pool(path, results):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "source_tag": r.source_tag,
                "alpha_idx": r.alpha_idx,
                "alpha_value": r.alpha_value,
                "beta": r.beta,
                "gamma_idx": r.gamma_idx,
                "gamma_value": r.gamma_value,
                "subset_indices": r.subset_indices,
                "selection_score": r.selection_score,
                "metadata": r.metadata,
            }) + "\n")


def dump_final_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_csv(path, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_run_readme(path, text):
    with open(path, "w") as f:
        f.write(text)
