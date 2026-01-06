from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize eval CSV.")
    parser.add_argument("--csv", required=True, help="Path to eval_*.csv")
    args = parser.parse_args()

    p = Path(args.csv)
    rows = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        raise SystemExit("No rows in CSV")

    def fnum(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    avg_mi = sum(fnum(r["must_include_score"]) for r in rows) / len(rows)
    avg_g = sum(fnum(r["grounding_score"]) for r in rows) / len(rows)
    total_viol = sum(int(r["must_not_include_violations"] or 0) for r in rows)

    print("Rows:", len(rows))
    print("Avg must_include_score:", round(avg_mi, 3))
    print("Avg grounding_score:", round(avg_g, 3))
    print("Total must_not_include_violations:", total_viol)


if __name__ == "__main__":
    main()
