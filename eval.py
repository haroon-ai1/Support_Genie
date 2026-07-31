"""Retrieval + routing evaluation for SupportGenie.

What this measures, and why each part exists:

  hit@k (chunk level)  The retrieved chunk must actually contain the answer
                       text, not merely come from the right file. Filename-level
                       scoring is close to meaningless with only three seed
                       documents — three slots, three files, and chance carries
                       most of the score.

  distractors          Six unrelated policy documents are indexed alongside the
                       seed set. Without them the index has nothing to be wrong
                       about and every number is inflated.

  abstain accuracy     Ten questions with no answer anywhere in the index. A
                       support bot that answers these is worse than useless, so
                       correctly falling below CONFIDENCE_THRESHOLD is scored
                       explicitly. "hard" negatives sit near an indexed topic
                       but ask for a fact the documents do not contain — that is
                       where a single cosine threshold is weakest, so they are
                       reported separately rather than hidden in the average.

  routing              Greetings must not be escalated to a human. Checked
                       against rag._is_smalltalk without calling the LLM, so the
                       whole eval runs offline and costs nothing.

Runs against a throwaway index in a temp directory, so it never touches the
live knowledge base and is reproducible from a clean checkout:

    python eval.py
"""
import json
import shutil
import tempfile
from pathlib import Path

from app import config

SEED_DIR = config.ROOT_DIR / "data" / "seed"
DISTRACTOR_DIR = config.ROOT_DIR / "data" / "eval" / "distractors"

# (question, expected source file, substring that must appear in a retrieved chunk)
POSITIVES = [
    # --- warranty_returns.txt ---
    ("How long is the warranty on products?", "warranty_returns.txt", "12-month standard warranty"),
    ("Does the warranty cover water damage?", "warranty_returns.txt", "water damage"),
    ("What is GenieCare+?", "warranty_returns.txt", "GenieCare+"),
    ("How do I claim warranty service?", "warranty_returns.txt", "start a claim online"),
    ("How many days do warranty repairs take?", "warranty_returns.txt", "7 to 10 working days"),
    ("Can I return a product I bought last week?", "warranty_returns.txt", "within 30 days of purchase"),
    ("Are earphones returnable?", "warranty_returns.txt", "earphones"),
    ("When will I get my refund after returning an item?", "warranty_returns.txt", "5 to 7 working days"),
    # paraphrased / customer-voice
    ("My phone stopped working after 8 months. Will you fix it for free?", "warranty_returns.txt", "12-month standard warranty"),
    ("I spilled tea on my laptop — is that covered?", "warranty_returns.txt", "water damage"),

    # --- shipping_payments.txt ---
    ("Do you deliver for free?", "shipping_payments.txt", "free for all orders above Rs. 5,000"),
    ("How long does delivery take to Lahore?", "shipping_payments.txt", "3 to 5 working days"),
    ("Is next day delivery available?", "shipping_payments.txt", "Express next-day delivery"),
    ("How can I track my order?", "shipping_payments.txt", "SMS and email"),
    ("Do you accept EasyPaisa?", "shipping_payments.txt", "EasyPaisa"),
    ("Is cash on delivery available?", "shipping_payments.txt", "Cash on delivery is available nationwide"),
    ("When does my order ship if I pay by bank transfer?", "shipping_payments.txt", "payment is verified"),
    ("Can I pay the rider when he arrives?", "shipping_payments.txt", "Cash on delivery"),
    ("How much extra to get it tomorrow in Islamabad?", "shipping_payments.txt", "Express next-day delivery"),

    # --- products_services.txt ---
    ("Do you sell Apple laptops?", "products_services.txt", "Samsung, Apple, Xiaomi"),
    ("Do you have a price match guarantee?", "products_services.txt", "lower advertised price"),
    ("Can I pay in installments?", "products_services.txt", "0% markup installment plans"),
    ("What are your store timings?", "products_services.txt", "11 AM to 10 PM"),
    ("Do you handle bulk corporate orders?", "products_services.txt", "corporate and bulk orders"),
    ("Which cities do you have stores in?", "products_services.txt", "Islamabad, Lahore, and Karachi"),
]

# (question, is_hard) — nothing in the index answers these.
NEGATIVES = [
    ("Do you offer a student discount?", False),
    ("What is your CEO's name?", False),
    ("Are you hiring software engineers?", False),
    ("What is the battery capacity of the Galaxy S23 Ultra?", False),
    ("Do you ship internationally to Dubai?", False),
    ("Can I book a repair appointment for a washing machine bought elsewhere?", False),
    # Hard: adjacent to an indexed topic, but the specific fact is absent.
    ("Is there a SupportGenie store in Peshawar?", True),
    ("Does the warranty cover the charger and cable separately?", True),
    ("Can I return an item after 45 days if it is unopened?", True),
    ("Is there an installment plan for orders under Rs. 10,000?", True),
]

SMALLTALK = [
    "hi", "hello", "hey there", "assalam o alaikum", "good morning",
    "thanks!", "thank you", "who are you", "what can you do", "bye",
]


def _build_isolated_kb():
    """Index seed + distractors into a temp location, leaving storage/ untouched."""
    tmp = Path(tempfile.mkdtemp(prefix="sg-eval-"))
    config.INDEX_PATH = tmp / "kb.faiss"
    config.CHUNKS_PATH = tmp / "chunks.json"
    config.STORAGE_DIR = tmp

    # Imported after the paths are patched so the KB picks them up.
    from app.ingest import KnowledgeBase

    kb = KnowledgeBase()
    n_seed = n_distract = 0
    for f in sorted(SEED_DIR.glob("*.txt")):
        n_seed += kb.add_document(f)
    for f in sorted(DISTRACTOR_DIR.glob("*.txt")):
        n_distract += kb.add_document(f)
    return kb, tmp, n_seed, n_distract


def main():
    from app.rag import _is_smalltalk

    kb, tmp, n_seed, n_distract = _build_isolated_kb()
    try:
        k = config.TOP_K
        threshold = config.CONFIDENCE_THRESHOLD

        # ---------- positives ----------
        hits = 0
        reciprocal_ranks = []
        grounded = 0
        pos_misses = []

        for question, src, needle in POSITIVES:
            results = kb.search(question, k=k)
            rank = None
            for i, r in enumerate(results):
                if r["source"] == src and needle.lower() in r["text"].lower():
                    rank = i + 1
                    break
            if rank:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                pos_misses.append((question, [(r["source"], round(r["score"], 3)) for r in results]))

            top = results[0]["score"] if results else 0.0
            if top >= threshold:
                grounded += 1

        n_pos = len(POSITIVES)
        hit_rate = hits / n_pos
        mrr = sum(reciprocal_ranks) / n_pos
        grounded_rate = grounded / n_pos

        # ---------- negatives ----------
        abstained = abstained_hard = n_hard = 0
        neg_failures = []
        for question, is_hard in NEGATIVES:
            results = kb.search(question, k=k)
            top = results[0]["score"] if results else 0.0
            ok = top < threshold
            if is_hard:
                n_hard += 1
                abstained_hard += int(ok)
            abstained += int(ok)
            if not ok:
                neg_failures.append((question, round(top, 3), is_hard))

        n_neg = len(NEGATIVES)
        abstain_rate = abstained / n_neg
        abstain_rate_hard = (abstained_hard / n_hard) if n_hard else 1.0

        # ---------- routing ----------
        routed = sum(1 for m in SMALLTALK if _is_smalltalk(m))
        smalltalk_rate = routed / len(SMALLTALK)

        # ---------- report ----------
        print(f"Index: {kb.index.ntotal} chunks "
              f"({n_seed} from seed, {n_distract} from distractors) | k={k} | threshold={threshold}")
        print()
        print(f"Positives ({n_pos})")
        print(f"  Hit@{k} (chunk level) : {hit_rate:.2%}  ({hits}/{n_pos})")
        print(f"  MRR                  : {mrr:.3f}")
        print(f"  Grounded (not escalated): {grounded_rate:.2%}")
        print()
        print(f"Negatives ({n_neg}) — correct abstention")
        print(f"  Overall              : {abstain_rate:.2%}  ({abstained}/{n_neg})")
        print(f"  Hard subset          : {abstain_rate_hard:.2%}  ({abstained_hard}/{n_hard})")
        print()
        print(f"Smalltalk routing ({len(SMALLTALK)})")
        print(f"  Not escalated        : {smalltalk_rate:.2%}")

        if pos_misses:
            print("\nRetrieval misses:")
            for q, got in pos_misses:
                print(f"  Q: {q}\n     retrieved: {got}")
        if neg_failures:
            print("\nAnswered when it should have abstained:")
            for q, score, is_hard in neg_failures:
                print(f"  [{'hard' if is_hard else 'easy'}] {q}  (top score {score} >= {threshold})")

        results_payload = {
            "k": k,
            "confidence_threshold": threshold,
            "chunks_indexed": int(kb.index.ntotal),
            "positives": {
                "n": n_pos,
                "hit_rate_at_k_chunk_level": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "grounded_rate": round(grounded_rate, 4),
            },
            "negatives": {
                "n": n_neg,
                "abstain_rate": round(abstain_rate, 4),
                "n_hard": n_hard,
                "abstain_rate_hard": round(abstain_rate_hard, 4),
            },
            "smalltalk": {
                "n": len(SMALLTALK),
                "not_escalated_rate": round(smalltalk_rate, 4),
            },
        }
        Path("eval_results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
        print("\nSaved to eval_results.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
