import uvicorn
import os
import sys

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting AI Engine via Python script...")
    # Use 'main:app' because main.py is in the root of ai-engine
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
