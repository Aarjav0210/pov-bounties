"""
Simplest possible API to test connection
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    print("✅ GET / called")
    return {"message": "Test API is working!"}

@app.get("/health")
def health():
    print("✅ GET /health called")
    return {"status": "healthy", "test": "working"}

@app.post("/test")
def test_post():
    print("✅ POST /test called")
    return {"message": "POST is working!"}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting TEST API on http://localhost:8000")
    print("Try these:")
    print("  - http://localhost:8000/")
    print("  - http://localhost:8000/health")
    print("\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

