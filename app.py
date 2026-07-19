"""Hugging Face Spaces entrypoint.

Gradio-SDK Spaces run `app.py` on port 7860. Our app is FastAPI, so this
shim just boots the existing FastAPI app via uvicorn.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)