"""Optional REST backend:  uvicorn api:app --reload   (docs at http://127.0.0.1:8000/docs)"""
from typing import List

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile

import pipeline as pl

app = FastAPI(title="VerbaTrack API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    try:
        return {"available": pl.list_models()}
    except pl.PipelineError as e:
        raise HTTPException(502, str(e))


@app.post("/process")
def process(file: UploadFile = File(...), glossary: str = Form("")):
    try:
        return pl.process_upload(file.file.read(), file.filename or "", glossary).to_dict()
    except pl.PipelineError as e:
        raise HTTPException(400, str(e))


@app.post("/ask")
def ask(question: str = Body(...), segments: List[dict] = Body(...)):
    """segments = the `segments` list returned by /process."""
    try:
        return pl.ask_meeting(question, segments).model_dump()
    except pl.PipelineError as e:
        raise HTTPException(400, str(e))
