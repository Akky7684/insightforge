"""API routes — dataset upload, chat, and system status endpoints."""

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from backend.app.config import get_settings
from backend.app.graph.supervisor import get_graph

router = APIRouter()


# --- Request & Response Models ---
class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or prompt")
    dataset_path: str = Field(..., description="Absolute or relative path to active dataset CSV")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Session / thread ID")
    user_id: Optional[str] = Field(default="default_user", description="User identifier")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    dataset_path: str


class DatasetMetadata(BaseModel):
    dataset_id: str
    filename: str
    file_path: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: dict
    preview: List[dict]


# --- Endpoints ---
@router.post("/upload", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Excel dataset file, cache it, and return structural metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx, .xls) files are supported")

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())[:8]
    safe_filename = f"{dataset_id}_{file.filename}"
    saved_path = upload_dir / safe_filename

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # If Excel, convert to CSV for consistent pandas reading in sandbox
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(saved_path)
            csv_path = saved_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            saved_path = csv_path

        # Read summary
        df = pd.read_csv(saved_path, encoding="latin1")
        row_count, col_count = df.shape
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        preview = df.head(5).fillna("").to_dict(orient="records")

        return DatasetMetadata(
            dataset_id=dataset_id,
            filename=file.filename,
            file_path=str(saved_path.resolve()),
            row_count=row_count,
            column_count=col_count,
            columns=columns,
            dtypes=dtypes,
            preview=preview,
        )
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded dataset: {e}")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(req: ChatRequest):
    """Execute a natural-language query over the dataset using the LangGraph agent."""
    if not os.path.exists(req.dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at: {req.dataset_path}")

    graph = get_graph()
    config = {"configurable": {"thread_id": req.session_id}}

    state_input = {
        "messages": [HumanMessage(content=req.message)],
        "dataset_id": Path(req.dataset_path).name,
        "dataset_path": str(Path(req.dataset_path).resolve()),
        "dataset_profile": None,
        "plan": [],
        "current_subtask_idx": 0,
        "rag_context": None,
        "pending_hitl_action": None,
        "session_id": req.session_id,
        "user_id": req.user_id,
    }

    try:
        final_state = graph.invoke(state_input, config=config)
        ai_messages = [m for m in final_state["messages"] if m.type == "ai"]
        latest_response = ai_messages[-1].content if ai_messages else "Analysis complete (no response generated)."

        return ChatResponse(
            response=latest_response,
            session_id=req.session_id,
            dataset_path=req.dataset_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {e}")


@router.get("/sample-datasets")
async def list_sample_datasets():
    """List bundled evaluation datasets in the data directory."""
    settings = get_settings()
    data_dir = Path(settings.data_dir)

    if not data_dir.exists():
        return []

    datasets = []
    for file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file, encoding="latin1", nrows=5)
            datasets.append({
                "name": file.name,
                "path": str(file.resolve()),
                "columns": df.columns.tolist(),
            })
        except Exception:
            continue

    return datasets


@router.get("/profile")
async def get_dataset_profile(dataset_path: str):
    """Generate or retrieve comprehensive statistical profile of a dataset."""
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at: {dataset_path}")

    try:
        from backend.app.graph.agents.profiler import profile_dataset
        profile = profile_dataset(dataset_path)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {e}")
