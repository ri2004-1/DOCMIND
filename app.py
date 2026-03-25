"""
DocMind - RAG Document Assistant
Uses Groq FREE API - no credit card, no payment ever
"""

import pickle, os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests

UPLOAD_DIR = Path("./uploads")
DB_FILE    = Path("./docstore.pkl")
UPLOAD_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.load()

    def save(self):
        with open(DB_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        if DB_FILE.exists():
            with open(DB_FILE, "rb") as f:
                self.chunks = pickle.load(f)

    def add(self, chunks):
        self.chunks.extend(chunks)
        self.save()

    def delete(self, source):
        self.chunks = [c for c in self.chunks if c["source"] != source]
        self.save()

    def search(self, query, k=4):
        if not self.chunks:
            return []
        query_words = set(query.lower().split())
        scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            if not query_words or not chunk_words:
                scores.append(0)
                continue
            intersection = len(query_words & chunk_words)
            union = len(query_words | chunk_words)
            scores.append(intersection / union if union > 0 else 0)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_k if scores[i] > 0]

    def get_sources(self):
        seen = {}
        for c in self.chunks:
            src = c["source"]
            seen[src] = seen.get(src, 0) + 1
        return [{"name": k, "chunks": v} for k, v in seen.items()]

db = SimpleVectorStore()

def split_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 30]

def extract_text(path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif ext == ".docx":
            import docx2txt
            return docx2txt.process(path)
        else:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
    except:
        return Path(path).read_text(encoding="utf-8", errors="ignore")

def ask_groq(prompt: str, temperature: float = 0.1) -> str:
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY not set. Add it in Render environment variables."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": temperature
            },
            timeout=30
        )
        if r.status_code != 200:
            return f"Groq error {r.status_code}: {r.text}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

app = FastAPI(title="DocMind")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text(encoding="utf-8")

@app.post("/upload")
async def upload(file: UploadFile = File(...), chunk_size: int = Form(500), chunk_overlap: int = Form(80)):
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(await file.read())
    text = extract_text(str(dest), file.filename)
    if not text.strip():
        return JSONResponse({"status": "error", "message": "Could not extract text"}, status_code=400)
    parts = split_text(text, chunk_size, chunk_overlap)
    chunks = [{"text": t, "source": file.filename, "chunk_idx": i} for i, t in enumerate(parts)]
    db.delete(file.filename)
    db.add(chunks) 
    return {"status": "indexed", "filename": file.filename, "chunks": len(chunks)}

@app.get("/documents")
async def documents():
    return {"documents": db.get_sources()}

@app.delete("/documents/{filename}")
async def delete(filename: str):
    db.delete(filename)
    f = UPLOAD_DIR / filename
    if f.exists(): f.unlink()
    return {"status": "deleted"}

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    system_prompt: Optional[str] = None

@app.post("/query")
async def query(req: QueryRequest):
    if not db.chunks:
        return {"answer": "No documents indexed yet. Please upload documents first.", "sources": []}
    results = db.search(req.question, req.top_k)
    if not results:
        return {"answer": "No relevant content found in your documents.", "sources": []}
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, Chunk #{c['chunk_idx']}]\n{c['text']}"
        for c, _ in results
    )
    system = req.system_prompt or "You are DocMind, a precise document assistant. Answer ONLY using the provided context. If not found, say so clearly."
    prompt = f"{system}\n\nDOCUMENT CONTEXT:\n{context}\n\nQUESTION: {req.question}\n\nANSWER:"
    answer = ask_groq(prompt, req.temperature)
    sources = [{"source": c["source"], "chunk_idx": c["chunk_idx"], "score": round(s, 3), "preview": c["text"][:180]+"..."} for c, s in results]
    return {"answer": answer, "sources": sources}

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(db.chunks), "api": "groq", "key_set": bool(GROQ_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  DocMind RAG Assistant")
    print("  Groq Free API | No Credit Card")
    print("="*50)
    print("  Open: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
