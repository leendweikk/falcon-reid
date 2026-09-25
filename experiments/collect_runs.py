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
        df = pd.read_csv(log)
        row = {"run": log.parent.name, "epochs_done": int(df.epoch.max()),
               "time_min": round(pd.to_numeric(df.seconds, errors="coerce").sum() / 60)}
        for col in ["mAP@10", "EMA_mAP@10"]:
            s = pd.to_numeric(df.get(col), errors="coerce")
            if s is not None and s.notna().any():
                i = s.idxmax()
                row[f"best {col}"] = round(float(s[i]), 4)
                row[f"at epoch ({col})"] = int(df.epoch[i])
                ev = df.loc[s.notna(), ["epoch"]].assign(v=s[s.notna()].round(4))
                row[f"{col} by epoch"] = " ".join(f"{int(e)}:{v}" for e, v in zip(ev.epoch, ev.v))
        rows.append(row)
    if rows:
        out.append("## Training runs (quick validation metric: cosine + flip, seed-42 split)\n")
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    for csv in sorted(RUNS.glob("*.csv")):
        out.append(f"## {csv.name}\n")
        out.append(pd.read_csv(csv).round(3).to_markdown(index=False) + "\n")

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
