"""Retrieval evaluation: hit rate@k and MRR over a held-out question set.

Each eval item maps a customer-style question to the source document that
contains its answer. A "hit" means the expected source appears in the top-k
retrieved chunks. Run AFTER seeding/ingesting the demo knowledge base:

    python eval.py
"""
import json

from app import config
from app.ingest import KnowledgeBase

EVAL_SET = [
    # warranty_returns.txt
    {"q": "How long is the warranty on products?", "src": "warranty_returns.txt"},
    {"q": "Does the warranty cover water damage?", "src": "warranty_returns.txt"},
    {"q": "What is VoltCare+?", "src": "warranty_returns.txt"},
    {"q": "How do I claim warranty service?", "src": "warranty_returns.txt"},
    {"q": "How many days do warranty repairs take?", "src": "warranty_returns.txt"},
    {"q": "Can I return a product I bought last week?", "src": "warranty_returns.txt"},
    {"q": "Are earphones returnable?", "src": "warranty_returns.txt"},
    {"q": "When will I get my refund after returning an item?", "src": "warranty_returns.txt"},
    # shipping_payments.txt
    {"q": "Do you deliver for free?", "src": "shipping_payments.txt"},
    {"q": "How long does delivery take to Lahore?", "src": "shipping_payments.txt"},
    {"q": "Is next day delivery available?", "src": "shipping_payments.txt"},
    {"q": "How can I track my order?", "src": "shipping_payments.txt"},
    {"q": "Do you accept EasyPaisa?", "src": "shipping_payments.txt"},
    {"q": "Is cash on delivery available?", "src": "shipping_payments.txt"},
    {"q": "When does my order ship if I pay by bank transfer?", "src": "shipping_payments.txt"},
    # products_services.txt
    {"q": "Do you sell Apple laptops?", "src": "products_services.txt"},
    {"q": "Do you have a price match guarantee?", "src": "products_services.txt"},
    {"q": "Can I pay in installments?", "src": "products_services.txt"},
    {"q": "What are your store timings?", "src": "products_services.txt"},
    {"q": "Do you handle bulk corporate orders?", "src": "products_services.txt"},
]


def main():
    kb = KnowledgeBase()
    if kb.index.ntotal == 0:
        print("Knowledge base is empty — run the app once (to seed) or ingest the seed docs first:")
        print("  python test_pipeline.py ingest data/seed/warranty_returns.txt   (and the other two)")
        return

    k = config.TOP_K
    hits, reciprocal_ranks, misses = 0, [], []

    for item in EVAL_SET:
        results = kb.search(item["q"], k=k)
        sources = [r["source"] for r in results]
        if item["src"] in sources:
            hits += 1
            reciprocal_ranks.append(1.0 / (sources.index(item["src"]) + 1))
        else:
            reciprocal_ranks.append(0.0)
            misses.append((item["q"], sources, [round(r["score"], 3) for r in results]))

    n = len(EVAL_SET)
    hit_rate = hits / n
    mrr = sum(reciprocal_ranks) / n

    print(f"Eval set: {n} questions | k = {k}")
    print(f"Hit rate@{k}: {hit_rate:.2%}  ({hits}/{n})")
    print(f"MRR:         {mrr:.3f}")

    if misses:
        print("\nMisses:")
        for q, srcs, scores in misses:
            print(f"  Q: {q}")
            print(f"     retrieved: {list(zip(srcs, scores))}")

    with open("eval_results.json", "w") as f:
        json.dump({"n": n, "k": k, "hit_rate": hit_rate, "mrr": mrr}, f, indent=2)
    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    main()
