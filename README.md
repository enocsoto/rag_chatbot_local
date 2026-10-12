# RAG y chatbot local

Lee documentos de `docs/`, los parte, pide embeddings a un servidor local y responde en streaming con lo que encontró.

El modelo no va dentro del repo. Tiene que haber un servidor OpenAI-compatible en `OLLAMA_BASE_URL` (Ollama en `http://127.0.0.1:11434/v1`). OpenCode usa ese mismo puerto.

## Cómo correrlo

Requiere Python 3.12 y Ollama.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
.venv/bin/python rag_local.py ingest docs
.venv/bin/python rag_local.py chat
```

En una Mac de 16 GB el modelo local realista es uno de unos 8B. DeepSeek-V4.1-Flash, GLM-5.3 y Kimi K3 no entran en esa memoria.

## Variables

| Variable | Uso |
| --- | --- |
| `OLLAMA_BASE_URL` | API compatible con OpenAI. Por defecto `http://127.0.0.1:11434/v1` |
| `OLLAMA_MODEL` | Modelo de chat. Por defecto `deepseek-r1:8b` |
| `OLLAMA_EMBED_MODEL` | Modelo de embeddings. Por defecto `nomic-embed-text` |
