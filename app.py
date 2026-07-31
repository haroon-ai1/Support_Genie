"""Local / Hugging Face Spaces entrypoint.

The app is FastAPI; this shim just boots it with uvicorn. Honours $PORT so the
same file works on Render (injects PORT), HF Spaces (expects 7860), and locally.
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,
    )
