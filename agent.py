"""Safe RAG Assistant — logique principale.

Pipeline d'une question :
    1. Garde-fou n°1 : règles (expressions régulières), rapide et gratuit.
    2. Garde-fou n°2 : un LLM classe la question avec une sortie structurée (Pydantic).
    3. Agent : le LLM décide d'appeler ses outils
         - retrieve_documents : recherche sémantique dans la base vectorielle (RAG)
         - calculator         : calculs exacts et sécurisés
       puis rédige la réponse finale à partir des résultats des outils.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from knowledge_base import DOCUMENTS

load_dotenv()  # lit la clé API dans le fichier .env (jamais écrite dans le code)

CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus-08-2024")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
TOP_K = 2
MAX_QUERY_CHARS = 1000


# ---------------------------------------------------------------------------
# Composants (créés à la première utilisation, remplaçables pour les tests)
# ---------------------------------------------------------------------------
_components: dict[str, Any] = {"llm": None, "embeddings": None, "store": None, "agent": None}


def configure(llm=None, embeddings=None) -> None:
    """Remplace le LLM et/ou le modèle d'embeddings (utile pour les tests)."""
    _components.update(llm=llm, embeddings=embeddings, store=None, agent=None)


def _check_api_key() -> None:
    if not os.getenv("COHERE_API_KEY"):
        raise RuntimeError(
            "COHERE_API_KEY manquante : copie .env.example en .env et ajoute ta clé Cohere."
        )


def get_llm():
    if _components["llm"] is None:
        _check_api_key()
        from langchain_cohere import ChatCohere

        _components["llm"] = ChatCohere(model=CHAT_MODEL, temperature=0.0)
    return _components["llm"]


def get_embeddings():
    if _components["embeddings"] is None:
        _check_api_key()
        from langchain_cohere import CohereEmbeddings

        _components["embeddings"] = CohereEmbeddings(model=EMBED_MODEL)
    return _components["embeddings"]


def get_vector_store() -> InMemoryVectorStore:
    """Base vectorielle : chaque document est transformé en embedding une seule fois."""
    if _components["store"] is None:
        _components["store"] = InMemoryVectorStore.from_texts(
            DOCUMENTS,
            get_embeddings(),
            metadatas=[{"source": f"doc-{i}"} for i in range(1, len(DOCUMENTS) + 1)],
        )
    return _components["store"]


# ---------------------------------------------------------------------------
# Outils de l'agent
# ---------------------------------------------------------------------------
@tool
def retrieve_documents(query: str) -> str:
    """Search the knowledge base about AI, machine learning, LLMs, RAG, embeddings,
    agents, AI ethics, guardrails and GPUs. Returns the most relevant passages with
    their source id."""
    results = get_vector_store().similarity_search_with_score(query, k=TOP_K)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(
        f"[{doc.metadata['source']}] (similarity {score:.2f}) {doc.page_content}"
        for doc, score in results
    )


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Évalue une expression arithmétique SANS eval() : seuls les nombres et les
    opérateurs + - * / // % ** sont autorisés (pas de code Python arbitraire)."""
    expr = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", ".")
    if len(expr) > 200:
        raise ValueError("expression too long")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("only numbers and + - * / // % ** are allowed")

    result = _eval(ast.parse(expr, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


@tool
def calculator(expression: str) -> str:
    """Compute an arithmetic expression exactly, e.g. '5814 * 6884' or '(3 + 4) ** 2'."""
    try:
        return str(safe_eval(expression))
    except Exception as e:  # noqa: BLE001 - le message d'erreur est renvoyé au LLM
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Garde-fous
# ---------------------------------------------------------------------------
class SafetyVerdict(BaseModel):
    """Sortie structurée du garde-fou LLM (validée par Pydantic)."""

    safe: bool = Field(description="True if the request can be answered safely")
    category: Literal[
        "none", "violence", "weapons", "self_harm", "illegal_activity", "hate", "other"
    ] = Field(description="Category of harm, 'none' if safe")
    reason: str = Field(description="One short sentence explaining the decision")


# Garde-fou n°1 : règles simples, ciblées pour limiter les faux positifs
UNSAFE_RULES = [
    (r"\b(suicide|suicidal|self[- ]?harm|kill (myself|me)|end my life)\b", "self_harm"),
    (
        r"\b(make|build|create|assemble|manufacture)\b.{0,40}"
        r"\b(bombs?|explosives?|weapons?|guns?|poisons?)\b",
        "weapons",
    ),
    (
        r"\b(kill|hurt|harm|poison|attack)\b.{0,20}\b(someone|somebody|people|a person|"
        r"him|her|them|my (wife|husband|boss|neighbou?r|friend|teacher|parents?))\b",
        "violence",
    ),
]


def rule_check(query: str) -> SafetyVerdict | None:
    """Renvoie un verdict 'dangereux' si une règle correspond, sinon None."""
    text = query.lower()
    for pattern, category in UNSAFE_RULES:
        if re.search(pattern, text):
            return SafetyVerdict(safe=False, category=category, reason="Matched a safety rule.")
    return None


SAFETY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a safety classifier for a general-purpose assistant. Decide whether the "
            "user request is safe to answer. Mark as UNSAFE only requests that seek help to "
            "harm people, create weapons or explosives, commit crimes, promote hate, or that "
            "indicate a risk of self-harm. Everyday, educational and technical questions are "
            "SAFE (e.g. 'how to kill a Linux process' is safe).",
        ),
        ("human", "{query}"),
    ]
)


def llm_check(query: str) -> SafetyVerdict:
    """Garde-fou n°2 : le LLM classe la question (sortie structurée Pydantic)."""
    classifier = SAFETY_PROMPT | get_llm().with_structured_output(SafetyVerdict)
    return classifier.invoke({"query": query})


def check_safety(query: str) -> SafetyVerdict:
    return rule_check(query) or llm_check(query)


REFUSALS = {
    "self_harm": (
        "It sounds like you might be going through something really hard. I can't help with "
        "this, but you don't have to face it alone: please talk to someone you trust or call a "
        "crisis line (in France: 3114, free and available 24/7)."
    ),
    "default": "I can't help with that request. Feel free to ask me something else!",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Safe RAG Assistant, a concise and helpful assistant.
- For questions about AI, machine learning, LLMs, RAG, embeddings, agents, AI ethics, \
guardrails or GPUs, ALWAYS call `retrieve_documents` first and base your answer on the \
returned passages. Cite the sources you used in brackets, e.g. [doc-4].
- If the passages do not answer the question, say that the knowledge base does not cover \
it, then give a short general answer.
- For any arithmetic, call `calculator` instead of computing mentally.
- For greetings and small talk, answer directly without tools.
- Answer in the same language as the user."""


def get_agent():
    if _components["agent"] is None:
        _components["agent"] = create_agent(
            model=get_llm(),
            tools=[retrieve_documents, calculator],
            system_prompt=SYSTEM_PROMPT,
        )
    return _components["agent"]


def _to_text(content: Any) -> str:
    """Le contenu d'un message peut être une chaîne ou une liste de blocs."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def answer(query: str) -> str:
    """Point d'entrée unique : garde-fous PUIS agent. Utilisé par l'API et l'interface."""
    query = (query or "").strip()
    if not query:
        return "Please type a question."
    if len(query) > MAX_QUERY_CHARS:
        return f"Your question is too long (maximum {MAX_QUERY_CHARS} characters)."

    try:
        verdict = check_safety(query)
    except RuntimeError as e:  # configuration manquante (clé API)
        return str(e)
    except Exception:  # noqa: BLE001 - en cas de doute, on refuse (fail closed)
        return "The safety check is temporarily unavailable, please try again later."
    if not verdict.safe:
        return REFUSALS.get(verdict.category, REFUSALS["default"])

    try:
        result = get_agent().invoke({"messages": [{"role": "user", "content": query}]})
    except Exception:  # noqa: BLE001
        return "Sorry, the assistant is temporarily unavailable. Please try again later."
    return _to_text(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Runnable exposé par LangServe (les garde-fous sont TOUJOURS appliqués)
# ---------------------------------------------------------------------------
class AgentInput(BaseModel):
    input: str = Field(description="The user's question")


def _invoke(data: Any) -> str:
    query = data.input if isinstance(data, AgentInput) else data["input"]
    return answer(query)


safe_agent = RunnableLambda(_invoke).with_types(input_type=AgentInput, output_type=str)


if __name__ == "__main__":
    print("Safe RAG Assistant (type 'exit' to quit)")
    while True:
        question = input("\nYour question: ")
        if question.lower() in {"exit", "quit", "q"}:
            break
        print("\nResponse:\n" + answer(question))
