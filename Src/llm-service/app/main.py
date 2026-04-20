from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.llms import get_llm_oss , get_llm_qwn , get_llm_llama

load_dotenv()



@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="LLM Service",
    description="Handles all LLM inference",
    version="0.1",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str
    context: str = ""
    model: str = "reasoning"  # "reasoning" | "arabic" | "assistant"


class GenerateResponse(BaseModel):
    answer: str
    model_used: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Generate a response from the requested model.
    """
    try:
        if req.model == "arabic":
            llm = get_llm_qwn()
        elif req.model == "assistant":
            llm = get_llm_oss()
        else:
            llm = get_llm_llama()

        full_prompt = f"{req.context}\n\n{req.prompt}" if req.context else req.prompt
        result = llm.invoke(full_prompt)

        return GenerateResponse(
            answer=result.content,
            model_used=req.model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)