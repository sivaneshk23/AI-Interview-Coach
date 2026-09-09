from fastapi import FastAPI

from app.routes import router


app = FastAPI(
    title="AI Interview Trainer Agent",
    description=(
        "An AI-powered interview training system "
        "using IBM watsonx foundation models."
    ),
    version="1.0.0",
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "AI Interview Trainer Agent",
        "status": "running",
        "version": "1.0.0",
    }