from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graph_registry import get_chat_graph
from app.guardrails import DISCLAIMER, STREAM_FALLBACK_REPLY

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    # "system" is deliberately not allowed: a client must never be able to
    # plant system-level turns into the graph state.
    role: Literal["user", "assistant"] = "user"
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=30)
    stream: bool = False
    # Deprecated: kept for request compatibility only. The verified PostVisit
    # token is the sole source of data scope; these fields are ignored.
    patient_id: str | None = None
    rekam_medis_id: str | None = None
    token: str | None = None


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    rekam_medis: list | None = None


async def _stream_reply(messages: list[dict], token: str | None = None):
    state = {
        "messages": messages,
        "patient_id": None,
        "rekam_medis_id": None,
        "token": token,
    }
    try:
        graph = get_chat_graph()
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if not node_output:
                    continue
                msgs = node_output.get("messages", [])
                if msgs:
                    chunk = msgs[-1]
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content
    except Exception:
        # Headers are already sent at this point; never let the client render
        # silence as a complete answer.
        yield STREAM_FALLBACK_REPLY
    yield f"\n\n{DISCLAIMER}"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=422,
            detail="Pesan terakhir harus dari pasien (role 'user').",
        )

    if request.stream:
        return StreamingResponse(
            _stream_reply(messages, request.token),
            media_type="text/plain",
        )

    try:
        result = get_chat_graph().invoke({
            "messages": messages,
            "patient_id": None,
            "rekam_medis_id": None,
            "token": request.token,
        })
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Layanan chat sedang bermasalah. Silakan coba lagi.",
        )

    last = result["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    return ChatResponse(
        content=f"{content}\n\n{DISCLAIMER}",
        rekam_medis=result.get("rekam_medis") or [],
    )
