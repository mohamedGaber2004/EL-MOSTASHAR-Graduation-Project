from fastapi import FastAPI
from src.routers.case_router import router as case_router

app = FastAPI(
    title="Legal AI System",
    description="AI-powered legal case analysis system",
    version="1.0.0"
)

# Include routers
app.include_router(case_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Legal AI System API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
