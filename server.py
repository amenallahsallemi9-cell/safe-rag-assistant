"""API REST de l'assistant (LangServe + FastAPI).

Lancement : python server.py
- API        : POST http://127.0.0.1:8000/agent/invoke   {"input": {"input": "What is RAG?"}}
- Playground : http://127.0.0.1:8000/agent/playground/
- Docs       : http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from langserve import add_routes

from agent import safe_agent

app = FastAPI(
    title="Safe RAG Assistant API",
    version="1.0.0",
    description="Guardrails + RAG agent (semantic search + calculator) powered by Cohere.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Le runnable exposé inclut les garde-fous : impossible de les contourner via l'API.
add_routes(app, safe_agent, path="/agent")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
