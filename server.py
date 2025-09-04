# server.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import sys, shutil

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ВАЖНО: тот же стек, что и в твоём MVP
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ---- НАСТРОЙКИ ----
BASE_DIR = Path(r"C:\qwen_mvp")  # наш проект
DOCS_DIR = Path(r"C:\mp_docs\MarketingPlatform-documentation\docs\user-guide")  # замени на свой путь к .md/.mdx
INDEX_DIR = BASE_DIR / "faiss_index"
REBUILD_INDEX = True  # ставь True, когда меняешь базу

# Лёгкая мультиязычная модель эмбеддингов (быстро и достаточно для MVP)
EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 800, 120, 5

# LLM в Ollama (мы ранее собрали qwen3b-mp)
LLM_MODEL, LLM_NUM_CTX = "qwen3b-mp", 2048

# ---- ВСПОМОГАТЕЛЬНО ----
def read_md_docs(root) -> list[Document]:
    root = Path(root)  # ← если вдруг пришла строка — станем Path
    if not root.exists():
        raise FileNotFoundError(f"Каталог не найден: {root}")
    docs: list[Document] = []
    for pat in ("*.md", "*.mdx"):
        for p in root.rglob(pat):
            try:
                docs.append(Document(page_content=p.read_text(encoding="utf-8", errors="ignore"),
                                     metadata={"source": str(p)}))
            except Exception as e:
                print(f"Пропуск {p}: {e}", file=sys.stderr)
    return docs

def split_docs(docs: list[Document]) -> list[Document]:
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
    body = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                          separators=["\n\n","\n"," ",""])
    out=[]
    for d in docs:
        try:
            for s in header.split_text(d.page_content):
                meta={**d.metadata, **s.metadata}
                for ch in body.split_text(s.page_content):
                    out.append(Document(page_content=ch, metadata=meta))
        except Exception:
            for ch in body.split_text(d.page_content):
                out.append(Document(page_content=ch, metadata=d.metadata))
    return out

def build_index() -> FAISS:
    # Эмбеддинги на CPU (надёжно для твоей связки); когда поставишь PyTorch nightly cu128 — вернём 'cuda'
    device = "cuda"
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            # под 8 ГБ VRAM начни так; если OOM — уменьши
            "batch_size": 128 if "MiniLM" in EMBED_MODEL else 16
        },
    )
    if REBUILD_INDEX and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    if not INDEX_DIR.exists():
        raw = read_md_docs(DOCS_DIR)
        if not raw:
            raise ValueError("Нет .md/.mdx документов")
        chunks = split_docs(raw)
        print(f"Документов: {len(raw)} | Чанков: {len(chunks)} | Emb: {EMBED_MODEL} @{device}")
        vs = FAISS.from_documents(chunks, emb)
        vs.save_local(str(INDEX_DIR))
        (BASE_DIR/"build_info.md").write_text(
            f"# RAG build\n- {datetime.now():%Y-%m-%d %H:%M:%S}\n- docs: {len(raw)}\n- chunks: {len(chunks)}\n"
            f"- device: {device}\n- emb: {EMBED_MODEL}\n- llm: {LLM_MODEL}\n", encoding="utf-8")
        return vs
    return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)

def make_qa(vs: FAISS) -> RetrievalQA:
    prompt = PromptTemplate(
        template=("Отвечай строго по контексту внутренней документации. Если ответа нет — скажи «Не нашёл в документации».\n\n"
                  "Контекст:\n{context}\n\nВопрос: {question}\n\nКраткий точный ответ. В конце перечисли источники (пути к файлам)."),
        input_variables=["context","question"])
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=LLM_NUM_CTX)
    retr = vs.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retr, chain_type="stuff",
                                       return_source_documents=True, chain_type_kwargs={"prompt": prompt})

# ---- FastAPI ----
app = FastAPI(title="Qwen RAG MVP")

# Разрешим запросы с твоего GitHub Pages
# Пример: https://<username>.github.io
ALLOWED_ORIGINS = [
    "https://github.com/george501",
    "https://github.com/george501/darts_and_shit/",  # если Pages как project site
    "http://localhost:5500",  # локальные тесты из Live Server (VS Code)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Инициализируем цепочку один раз
VSTORE = build_index()
QA = make_qa(VSTORE)

class AskIn(BaseModel):
    question: str

class AskOut(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    res = QA.invoke({"query": payload.question})
    answer = (res.get("result") or "").strip()
    srcs = []
    for d in (res.get("source_documents") or []):
        path = d.metadata.get("source","")
        if path and path not in srcs:
            srcs.append(path)
    return AskOut(answer=answer, sources=srcs)

# uvicorn запускать из PowerShell:
# uvicorn server:app --host 0.0.0.0 --port 8000
