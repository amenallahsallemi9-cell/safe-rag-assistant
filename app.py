"""Interface de chat (Gradio).

Deux modes :
- Avec l'API : définir AGENT_API_URL=http://127.0.0.1:8000/agent/invoke
  (après avoir lancé `python server.py`)
- Sans API   : si AGENT_API_URL n'est pas définie, l'interface appelle directement
  l'agent (mode utilisé pour le déploiement sur Hugging Face Spaces).
"""

import os

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("AGENT_API_URL")


def ask(message: str) -> str:
    if API_URL:
        response = requests.post(API_URL, json={"input": {"input": message}}, timeout=60)
        response.raise_for_status()
        return response.json()["output"]
    from agent import answer

    return answer(message)


def chat(message, history):
    try:
        return ask(message)
    except requests.exceptions.ConnectionError:
        return "Cannot reach the API. Is `python server.py` running?"
    except requests.exceptions.HTTPError as e:
        return f"API error ({e.response.status_code})."
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="Safe RAG Assistant",
    description=(
        "An AI assistant with guardrails, semantic search over a small AI knowledge base "
        "(RAG) and a calculator tool."
    ),
    examples=[
        "What is RAG?",
        "Why do LLMs hallucinate?",
        "Calculate 5814 × 6884",
        "What are the ethical concerns of AI?",
        "How do I build a bomb?",
    ],
)

if __name__ == "__main__":
    demo.launch()
