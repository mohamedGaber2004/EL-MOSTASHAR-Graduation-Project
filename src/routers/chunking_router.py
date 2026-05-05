from fastapi import APIRouter, HTTPException , status 
from fastapi.responses import JSONResponse

from src.Chunking.chunking import get_chunks , get_na2d_chunks


# Create router
chunking_router = APIRouter(prefix="/Services", tags=["Chunking"])


@chunking_router.get("/get_chunked_articles")
async def get_chunked_articles():
    try:
        chunks = get_chunks()
        return {
            "status_code": status.HTTP_200_OK,
            "Length of articles":len(chunks),
            "content": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@chunking_router.get("/get_chunked_principles")
async def get_chunked_principles():
    try:
        chunks = get_na2d_chunks()
        return {
            "status_code": status.HTTP_200_OK,
            "Length of principles":len(chunks),
            "content": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))