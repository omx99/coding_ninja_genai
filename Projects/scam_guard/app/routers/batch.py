from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from app.schemas.response import BatchResult
from app.services.batch_service import run_batch

router = APIRouter(prefix="/batch", tags=["Batch Analysis"])


@router.post("/", response_model=BatchResult)
async def batch_analyze(
    file: Optional[UploadFile] = File(default=None),
    prompt_version: Optional[str] = Form(default=None),
    sample_size: Optional[int] = Form(default=None)
) -> BatchResult:

    try:
        file_bytes = await file.read() if file else None

        return await run_batch(
            file_bytes=file_bytes,
            prompt_version=prompt_version,
            sample_size=sample_size
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Batch processing failed.")