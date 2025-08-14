# eval/precision_at_k.py
import os, sys
# ensure project root is on sys.path so top-level packages can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json, os, time, statistics
from pathlib import Path

# Import your existing retriever (same one Streamlit uses)
from embeddings.openai_embed_and_store import query_chroma_by_embedding

HERE = Path(__file__).parent
TESTS_PATH = HERE / "tests.json"

def load_tests():
    if TESTS_PATH.exists():
        with open(TESTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback defaults if tests.json is missing
    return [
        {"q": "Who are the parties in the agreement?", "expected": ["sample.txt.txt"]},
        {"q": "What is the payment amount?", "expected": ["sample.txt.txt"]},
        {"q": "What is the termination notice period?", "expected": ["sample.txt.txt"]},
    ]

def topk_filenames(result, k):
    metas = result.get("metadatas", [[]])[0]
    return [m.get("filename") for m in metas][:k] if metas else []

def run(k_values=(1,3,5)):
    tests = load_tests()
    if not tests:
        print("No tests found.")
        return

    # Make sure we query up to the largest k we plan to measure
    max_k = max(k_values)

    perq_times = []
    per_k_hits = {k: 0 for k in k_values}

    print(f"Found {len(tests)} test queries.")
    for t in tests:
        q = t["q"]
        expected = set(t["expected"])

        t0 = time.time()
        res = query_chroma_by_embedding(q, k=max_k)
        dt = time.time() - t0
        perq_times.append(dt)

        print(f"\nQ: {q}")
        print(f"  retrieval time: {dt:.3f}s")

        for k in k_values:
            topk = topk_filenames(res, k)
            hit = any(fn in expected for fn in topk)
            per_k_hits[k] += int(hit)
            print(f"  top-{k} files: {topk} -> {'HIT' if hit else 'MISS'}")

    print("\n===== SUMMARY =====")
    n = len(tests)
    for k in k_values:
        precision = per_k_hits[k] / n
        print(f"Precision@{k}: {per_k_hits[k]}/{n} = {precision:.2f}")
    avg = statistics.mean(perq_times)
    p95 = statistics.quantiles(perq_times, n=20)[-1] if len(perq_times) >= 2 else avg
    print(f"Avg retrieval time: {avg:.3f}s   (p95 ≈ {p95:.3f}s)")

if __name__ == "__main__":
    run()
