from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def load_eval_csv(path: str | Path) -> Dict[str, dict]:
    """
    Load eval CSV keyed by question id.
    """
    p = Path(path)
    rows: Dict[str, dict] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"]
            rows[qid] = row
    if not rows:
        raise ValueError(f"No rows found in {p}")
    return rows


def fnum(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compare_runs(old_csv: str | Path, new_csv: str | Path) -> Dict[str, any]:
    old = load_eval_csv(old_csv)
    new = load_eval_csv(new_csv)

    common_ids = sorted(set(old) & set(new))
    if not common_ids:
        raise ValueError("No overlapping question IDs between runs")

    deltas: List[dict] = []

    for qid in common_ids:
        o = old[qid]
        n = new[qid]

        d = {
            "id": qid,
            "question": n.get("question", ""),
            "delta_must_include": fnum(n["must_include_score"]) - fnum(o["must_include_score"]),
            "delta_grounding": fnum(n["grounding_score"]) - fnum(o["grounding_score"]),
            "delta_violations": int(n["must_not_include_violations"]) - int(o["must_not_include_violations"]),
        }
        deltas.append(d)

    def avg(key: str) -> float:
        return sum(d[key] for d in deltas) / len(deltas)

    summary = {
        "count": len(deltas),
        "avg_delta_must_include": round(avg("delta_must_include"), 3),
        "avg_delta_grounding": round(avg("delta_grounding"), 3),
        "total_new_violations": sum(max(0, d["delta_violations"]) for d in deltas),
    }

    return {
        "summary": summary,
        "deltas": deltas,
    }


def write_regression_csv(deltas: List[dict], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "delta_must_include",
        "delta_grounding",
        "delta_violations",
    ]

    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in deltas:
            w.writerow(d)


def write_markdown_report(
    old_csv: str | Path,
    new_csv: str | Path,
    summary: dict,
    deltas: List[dict],
    out_path: str | Path,
    top_n: int = 5,
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    worst = sorted(
        deltas,
        key=lambda d: (d["delta_must_include"] + d["delta_grounding"]),
    )[:top_n]

    with p.open("w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Regression Report\n\n")
        f.write(f"**Old run:** `{old_csv}`  \n")
        f.write(f"**New run:** `{new_csv}`\n\n")

        f.write("## Summary\n")
        f.write(f"- Questions compared: {summary['count']}\n")
        f.write(f"- Avg must_include delta: {summary['avg_delta_must_include']}\n")
        f.write(f"- Avg grounding delta: {summary['avg_delta_grounding']}\n")
        f.write(f"- New violations introduced: {summary['total_new_violations']}\n\n")

        f.write("## Worst Regressions\n")
        for d in worst:
            f.write(f"- **{d['id']}**: must_include {d['delta_must_include']:+.3f}, "
                    f"grounding {d['delta_grounding']:+.3f}, "
                    f"violations {d['delta_violations']:+d}\n")
            f.write(f"  - {d['question']}\n")


def main() -> None:
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Compare two eval CSVs and report regressions.")
    parser.add_argument("--old", required=True, help="Older eval_*.csv")
    parser.add_argument("--new", required=True, help="Newer eval_*.csv")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    result = compare_runs(args.old, args.new)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir)

    csv_path = outdir / f"regression_{ts}.csv"
    md_path = outdir / f"regression_{ts}.md"

    write_regression_csv(result["deltas"], csv_path)
    write_markdown_report(
        args.old,
        args.new,
        result["summary"],
        result["deltas"],
        md_path,
    )

    print(f"Wrote:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
