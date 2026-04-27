# Import necessary modules
import uvicorn
from fastapi import FastAPI

# Initialize the FastAPI app
app = FastAPI()

# Define a simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Graduation Project API!"}

# Import the case router
from src.routers.case_router import router as case_router

# Include the case router in the app
app.include_router(case_router)

# Entry point for running the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)