import argparse
import json
import os

import numpy as np


def flatten(values):
    """Flatten nested lists of values."""
    return [v for sublist in values for v in (sublist if isinstance(sublist, list) else [sublist])]


def compute_stats(values):
    arr = np.array(values)
    return [
        f"{np.mean(arr):.4f}",
        f"{np.std(arr):.4f}",
        f"{np.percentile(arr, 25):.4f}",
        f"{np.percentile(arr, 50):.4f}",
        f"{np.percentile(arr, 75):.4f}",
    ]


def main(data_path, max_rows):
    json_file = "tote_stats_summary.json"
    json_path = os.path.join(data_path, json_file)

    with open(json_path) as f:
        data = json.load(f)

    gcus = []
    transfers = []
    ejections = []

    for k, v in data.items():
        if not k.isdigit():
            continue  # Skip mean_* keys
        gcus.extend(flatten(v["gcus"]))
        transfers.extend(flatten(v["obj_transfers"]))
        ejections.extend(flatten(v["source_ejections"]))

    print(f"Total records found: {len(gcus)}")

    if max_rows is not None:
        gcus = gcus[:max_rows]
        transfers = transfers[:max_rows]
        ejections = ejections[:max_rows]

    gcus_stats = compute_stats(gcus)
    transfers_stats = compute_stats(transfers)
    ejections_stats = compute_stats(ejections)

    # Print header for Google Sheets
    print("Header\tGCUs\t\t\t\t\tObjects transferred\t\t\t\t\tSource tote ejections\t\t\t\t\t")
    print("\tMean\tStdev\tp25\tp50\tp75\tMean\tStdev\tp25\tp50\tp75\tMean\tStdev\tp25\tp50\tp75")

    # Print values, with a label at the front
    print("Base " + " ".join(gcus_stats + transfers_stats + ejections_stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize metrics for Google Sheets.")
    parser.add_argument("--data_path", type=str, required=True, help="Root path to the saved container data.")
    parser.add_argument("--max_rows", type=int, default=None, help="Number of rows to read (default: all).")
    args = parser.parse_args()

    main(args.data_path, args.max_rows)
