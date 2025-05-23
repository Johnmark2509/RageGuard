import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # fallback to 8000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)