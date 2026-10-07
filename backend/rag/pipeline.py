import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph

from embeddings.embedder import embed_text
from vectorstore.chroma_store import create_collection, search

from auth.users import (
    get_user,
    get_allowed_departments
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class RagState(TypedDict, total=False):
    question: str
    user_id: str
    user: dict
    allowed_departments: list[str]
    top_k: int
    matches: list[dict]
    context: str
    answer: str


def retrieve(state: RagState) -> RagState:
    allowed_departments = state.get("allowed_departments", [])

    # Fail closed if user has no permissions
    if not allowed_departments:
        return {"matches": []}

    # Generate query embedding
    query_embedding = embed_text([state["question"]])[0]

    # Search only authorized departments
    results = search(
        create_collection(),
        query_embedding,
        state.get("top_k", 4),
        allowed_departments=allowed_departments
    )

    # Format retrieved chunks
    matches = [
        {
            "text": document,
            "citation": metadata,
            "distance": distance
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

    return {"matches": matches}


def augment(state: RagState) -> RagState:
    matches = state.get("matches", [])

    if not matches:
        return {
            "context": "",
            "answer": (
                "The uploaded documents do not contain "
                "this information in your authorized "
                "knowledge base."
            )
        }

    context_parts = []

    for i, chunk in enumerate(matches, start=1):
        citation = chunk.get("citation", {})
        source = citation.get("source", "Unknown source")
        page = citation.get("page", "Unknown")

        context_parts.append(
            f"[Source {i}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return {
        "context": context
    }

def generate(state: RagState) -> RagState:
    if not state.get("matches"):
        return {
            "answer": state.get(
                "answer",
                "The uploaded documents do not contain "
                "this information in your authorized knowledge base."
            )
        }

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in the backend .env file."
        )

    prompt = f"""Answer only from the supplied document context.
                If it does not support an answer, say exactly:
                'The uploaded documents do not contain this information.'
                Cite each factual claim with [Source N].

Context:
{state['context']}

Question: {state['question']}"""

    response = Groq(api_key=api_key).chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content or ""
    }


def identify_user(state: RagState) -> RagState:
    user_id = state.get("user_id")

    user = get_user(user_id)

    if user is None:
        return {
            "user": {},
            "allowed_departments": []
        }

    allowed_departments = get_allowed_departments(user)

    return {
        "user": user,
        "allowed_departments": allowed_departments
    }


def build_rag_graph():
    workflow = StateGraph(RagState)

    workflow.add_node("identify_user", identify_user)
    workflow.add_node("retrieve_authorized", retrieve)
    workflow.add_node("augment", augment)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "identify_user")
    workflow.add_edge("identify_user", "retrieve_authorized")
    workflow.add_edge("retrieve_authorized", "augment")
    workflow.add_edge("augment", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


rag_graph = build_rag_graph()
