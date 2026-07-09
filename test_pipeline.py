"""Night 1 goal: run this end-to-end in the terminal.

Usage:
    python test_pipeline.py ingest data/uploads/faq.pdf   # ingest a document
    python test_pipeline.py sources                        # list what's indexed
    python test_pipeline.py chat                           # interactive Q&A loop
"""
import sys

from app.ingest import KnowledgeBase
from app.rag import answer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]
    kb = KnowledgeBase()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python test_pipeline.py ingest <path-to-file>")
            return
        n = kb.add_document(sys.argv[2])
        print(f"Ingested {n} chunks. Total in index: {kb.index.ntotal}")

    elif command == "sources":
        counts = kb.sources()
        if not counts:
            print("Knowledge base is empty. Ingest a document first.")
            return
        for source, n in counts.items():
            print(f"  {source}: {n} chunks")

    elif command == "chat":
        if kb.index.ntotal == 0:
            print("Knowledge base is empty. Ingest a document first.")
            return
        print("Chat with your knowledge base (Ctrl+C or 'quit' to exit)\n")
        while True:
            try:
                q = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q or q.lower() in {"quit", "exit"}:
                break
            result = answer(kb, q)
            print(f"\nBot: {result['answer']}")
            if result["handoff"]:
                print(f"     [handoff triggered — confidence {result['confidence']}]")
            else:
                srcs = ", ".join(
                    f"{s['source']} ({s['score']})" for s in result["sources"]
                )
                print(f"     [confidence {result['confidence']} | sources: {srcs}]")
            print()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
