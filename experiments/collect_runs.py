"""
Collects every experiment result on disk into one file, so EXPERIMENTS.md is written from real numbers.
  python experiments/collect_runs.py      -> docs/experiments_raw.md
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def main():
    out = ["# Raw experiment results (auto-collected)\n"]

    rows = []
    for log in sorted(RUNS.glob("*/log.csv")):
        try:
            df = pd.read_csv(log)
        except Exception as e:                       # unreadable/empty log: note it, keep going
            rows.append({"run": log.parent.name, "note": f"unreadable: {e}"})
            continue
        row = {"run": log.parent.name, "columns": " ".join(df.columns) if "epoch" not in df.columns else ""}
        if "epoch" in df.columns:
            row["epochs_done"] = int(pd.to_numeric(df.epoch, errors="coerce").max())
        if "seconds" in df.columns:
            row["time_min"] = round(pd.to_numeric(df.seconds, errors="coerce").sum() / 60)
        for col in [c for c in df.columns if "mAP" in c]:
            s = pd.to_numeric(df[col], errors="coerce")
            if not s.notna().any():
                continue
            i = s.idxmax()
            ep = df.epoch if "epoch" in df.columns else pd.Series(range(1, len(df) + 1))
            row[f"best {col}"] = round(float(s[i]), 4)
            row[f"at epoch ({col})"] = int(ep[i])
            row[f"{col} by epoch"] = " ".join(f"{int(e)}:{round(float(v), 4)}" for e, v in zip(ep[s.notna()], s[s.notna()]))
        rows.append(row)
    if rows:
        out.append("## Training runs (quick validation metric: cosine + flip, seed-42 split)\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    for csv in sorted(RUNS.glob("*.csv")):
        out.append(f"## {csv.name}\n")
        try:
            out.append(pd.read_csv(csv).round(3).to_markdown(index=False) + "\n")
        except Exception as e:
            out.append(f"(unreadable: {e})\n")

    for csv in sorted(RUNS.glob("*/threshold_study.csv")) + sorted(RUNS.glob("*/*/threshold_study.csv")):
        out.append(f"## {csv.relative_to(RUNS)}\n")
        out.append(pd.read_csv(csv).round(4).to_markdown(index=False) + "\n")

    logs = sorted((RUNS / "queue_logs").glob("*.txt")) if (RUNS / "queue_logs").exists() else []
    for log in logs:
        raw = log.read_bytes()                        # PowerShell 5 Tee-Object writes UTF-16
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")
        lines = [l for l in text.splitlines() if l.strip()]
        out.append(f"## queue log: {log.stem} (last lines)\n```\n" + "\n".join(lines[-12:]) + "\n```\n")

    dst = ROOT / "docs" / "experiments_raw.md"
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"saved -> {dst}  ({len(rows)} training runs)")


if __name__ == "__main__":
    main()
