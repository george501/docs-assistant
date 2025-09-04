from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys, shutil, torch
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

DOCS_DIR = r"C:\mp_docs\MarketingPlatform-documentation\docs\user-guide"  # замени на свой путь к .md/.mdx
INDEX_DIR = "faiss_index"
REBUILD_INDEX = True
EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 800, 120, 5
LLM_MODEL, LLM_NUM_CTX = "qwen3b-mp", 2048

def read_md_docs(root):
    root = Path(root); docs=[]
    if not root.exists(): raise FileNotFoundError(f"Каталог не найден: {root}")
    for pat in ("*.md","*.mdx"):
        for p in root.rglob(pat):
            try: docs.append(Document(page_content=p.read_text(encoding="utf-8", errors="ignore"), metadata={"source": str(p)}))
            except Exception as e: print(f"Пропуск {p}: {e}", file=sys.stderr)
    return docs

def split_docs(docs):
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
    body = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n","\n"," ",""])
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
    if REBUILD_INDEX and Path(INDEX_DIR).exists(): shutil.rmtree(INDEX_DIR)
    if not Path(INDEX_DIR).exists():
        raw = read_md_docs(DOCS_DIR); 
        if not raw: raise ValueError("Нет .md/.mdx документов")
        chunks = split_docs(raw)
        print(f"Документов: {len(raw)} | Чанков: {len(chunks)} | Emb: {EMBED_MODEL} @{device}")
        vs = FAISS.from_documents(chunks, emb); vs.save_local(INDEX_DIR)
        Path("build_info.md").write_text(f"# RAG build\n- {datetime.now():%Y-%m-%d %H:%M:%S}\n- docs: {len(raw)}\n- chunks: {len(chunks)}\n- device: {device}\n- emb: {EMBED_MODEL}\n- llm: {LLM_MODEL}\n", encoding="utf-8")
        return vs
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

def make_qa(vs):
    prompt = PromptTemplate(
        template=("Отвечай строго по контексту внутренней документации. Если ответа нет — скажи «Не нашёл в документации».\n\n"
                  "Контекст:\n{context}\n\nВопрос: {question}\n\nКраткий точный ответ. В конце перечисли источники (пути к файлам)."),
        input_variables=["context","question"])
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1, num_ctx=LLM_NUM_CTX)
    retr = vs.as_retriever(search_kwargs={"k": TOP_K})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retr, chain_type="stuff", return_source_documents=True, chain_type_kwargs={"prompt": prompt})

def show_answer(res):
    print("\n=== ОТВЕТ ===\n" + res.get("result","").strip())
    print("\nИСТОЧНИКИ:")
    seen=set()
    for d in (res.get("source_documents") or []):
        s=d.metadata.get("source",""); 
        if s and s not in seen: print("—", s); seen.add(s)

if __name__=="__main__":
    vs = build_index(); qa = make_qa(vs)
    print("Готово. Введите вопрос (exit для выхода).")
    while True:
        q = input("> ").strip()
        if q.lower() in ("exit","выход"): break
        if not q: continue
        res = qa.invoke({"query": q}); show_answer(res)