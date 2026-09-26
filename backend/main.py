"""
DermaAI Demo Backend Entrypoint.
Re-exports the production FastAPI app from backend.app.main.
"""

from backend.app.main import app

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=False)
