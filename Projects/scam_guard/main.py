# rest endpoint ??


from fastapi import FastAPI
from app.core.config import get_setings
app = FastAPI()
from app.routers import batch
settings = get_setings()

app.include_router(batch.router)


@app.get("/",tags=["health"])
def home():
    from app.prompts.registry import prompt_registry
    return {
        "default_version": settings.default_prompt_version,
        "available_versions": prompt_registry.list_versions()
    }

@app.get("/version",tags=["health"])
def list_prompt_versions():
    from app.prompts.registry import prompt_registry
    return {
        "default_version": settings.default_prompt_version,
        "available_versions": prompt_registry.list_versions()
    }



