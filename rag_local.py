"""RAG local: documentos, embeddings y respuesta en streaming."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / ".rag" / "index.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Respondes en español usando solo este contexto. "
            "Si no alcanza, dilo. Contexto:\n{contexto}",
        ),
        ("human", "{pregunta}"),
    ]
)


def _settings() -> tuple[str, str, str]:
    load_dotenv()
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    chat_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return base, chat_model, embed_model


def _native_base(openai_base: str) -> str:
    if openai_base.endswith("/v1"):
        return openai_base[: -len("/v1")]
    return openai_base


def _chunks(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE].strip()
        if piece:
            pieces.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return pieces


def _read_docs(folder: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        for piece in _chunks(text):
            rows.append((str(path.relative_to(folder)), piece))
    return rows


async def _embed(base: str, model: str, text: str) -> list[float]:
    url = f"{_native_base(base)}/api/embed"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json={"model": model, "input": text})
        response.raise_for_status()
        vectors = response.json().get("embeddings") or []
    if not vectors:
        raise RuntimeError("El servidor no devolvió embeddings.")
    return vectors[0]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _search(index: dict, query: list[float], k: int = 4) -> str:
    ranked = sorted(
        index["chunks"],
        key=lambda item: _cosine(query, item["embedding"]),
        reverse=True,
    )
    blocks = [f"[{item['source']}]\n{item['text']}" for item in ranked[:k]]
    return "\n\n".join(blocks)


async def ingest(folder: Path) -> None:
    base, _chat_model, embed_model = _settings()
    rows = _read_docs(folder)
    if not rows:
        raise SystemExit(f"No hay .md ni .txt en {folder}")
    chunks = []
    for source, text in rows:
        chunks.append(
            {
                "source": source,
                "text": text,
                "embedding": await _embed(base, embed_model, text),
            }
        )
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps({"chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Indexados {len(chunks)} fragmentos en {INDEX_PATH}")


async def chat() -> None:
    if not INDEX_PATH.is_file():
        raise SystemExit("Falta el índice. Corre primero: python rag_local.py ingest docs")
    base, chat_model, embed_model = _settings()
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    model = ChatOpenAI(
        model_name=chat_model,
        temperature=0.2,
        base_url=base,
        api_key="ollama",
    )
    chain = prompt | model | StrOutputParser()
    print("Escribe la pregunta. Vacío para salir.")
    while True:
        question = input("> ").strip()
        if not question:
            return
        context = _search(index, await _embed(base, embed_model, question))
        async for token in chain.astream({"contexto": context, "pregunta": question}):
            print(token, end="", flush=True)
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("ingest", "chat"))
    parser.add_argument("folder", nargs="?", default="docs")
    args = parser.parse_args()
    if args.command == "ingest":
        asyncio.run(ingest(Path(args.folder)))
        return
    asyncio.run(chat())


if __name__ == "__main__":
    main()
