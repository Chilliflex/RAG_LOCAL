"""Phase 6: run ingest + eval of the golden set in both modes and write data/eval_compare.json.
Needs MISTRAL_API_KEY and GOOGLE_API_KEY for the cloud half. Each mode uses its own data folder, so nothing is overwritten.

  .venv\\Scripts\\python scripts\\compare_modes.py [--skip-local]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag import cloud  # noqa: E402

GOLDEN = ROOT / "samples" / "golden.json"


def run(provider: str) -> dict:
    env = {**os.environ, "PROVIDER": provider}
    files = sorted({g["file"] for g in json.loads(GOLDEN.read_text(encoding="utf-8"))})
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "rag", "ingest", *(str(ROOT / "samples" / f) for f in files)], check=True, env=env)
    ingest_seconds = round(time.time() - t0)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S")
    out = subprocess.run([sys.executable, "-m", "rag", "eval"], check=True, env=env, capture_output=True, text=True, encoding="utf-8").stdout
    result = json.loads(out[out.index("{"):])
    result["ingest_seconds_this_run"] = ingest_seconds  # 0-ish when the documents were already ingested
    if provider == "cloud":
        log = ROOT / "data-cloud" / "api_usage.jsonl"
        entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        result["cost"] = {"ingest_and_earlier": cloud.cost([e for e in entries if e["time"] < t1]),
                          "questions": cloud.cost([e for e in entries if e["time"] >= t1])}
    return result


def main():
    results = {}
    if "--skip-local" not in sys.argv:
        results["local"] = run("local")
    results["cloud"] = run("cloud")
    cats = sorted({c for r in results.values() for c in r["by_category"]})
    table = [{"category": c, **{p: f"{r['by_category'][c]['correct']}/{r['by_category'][c]['questions']}" for p, r in results.items() if c in r["by_category"]}}
             for c in cats]
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "modes": results, "accuracy_by_category": table}
    path = ROOT / "data" / "eval_compare.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'category':14}" + "".join(f"{p:>10}" for p in results))
    for row in table:
        print(f"{row['category']:14}" + "".join(f"{row.get(p, '-'):>10}" for p in results))
    for p, r in results.items():
        print(f"{p}: accuracy {r['accuracy']}, recall@5 {r['recall@5']}, {sum(c['seconds'] for c in r['by_category'].values()):.0f} s for {r['questions']} questions"
              + (f", cost of the questions ${r['cost']['questions']['usd']} (ingest and earlier ${r['cost']['ingest_and_earlier']['usd']})" if "cost" in r else ""))
    print("wrote", path)


if __name__ == "__main__":
    main()
