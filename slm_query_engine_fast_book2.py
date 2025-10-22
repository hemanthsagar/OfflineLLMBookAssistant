"""
Offline Small Language Model (SLM) Q&A Engine
---------------------------------------------
Optimized for:
  ✅ RTX 3050 (4GB VRAM)
  ✅ Fast response (1–2s)
  ✅ Local documents
"""

# === Imports ===
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from functools import lru_cache
import os

# === Configuration ===
BOOK_PATH = "Cracking-the-Coding-Interview.txt"           # path to your text file
INDEX_PATH = "cache/book_index"    # folder for FAISS persistence
MODEL_NAME = "microsoft/phi-2"   # smaller, faster than phi-2
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 400
TOP_K = 3
MAX_TOKENS = 1000
TEMPERATURE = 0.05                 # factual + deterministic


# === Step 1: Load and split text ===
print("📚 Loading and chunking text...")
with open(BOOK_PATH, "r", encoding="utf-8") as f:
    book_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
docs = [Document(page_content=chunk) for chunk in splitter.split_text(book_text)]
print(f"✅ Loaded {len(docs)} text chunks.")


# === Step 2: Build or load FAISS index ===
print("⚙️ Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if os.path.exists(INDEX_PATH):
    print("📦 Loading existing FAISS index...")
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("🧠 Creating new FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    db.save_local(INDEX_PATH)
print("✅ FAISS index ready.")


# === Step 3: Load Quantized Model (4-bit for speed + low VRAM) ===
print("🚀 Loading Phi model in 4-bit quantized mode...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_cfg
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Warm up GPU to avoid first-query delay
_ = generator("hello", max_new_tokens=1)


# === Step 4: Cached retrieval helper ===
@lru_cache(maxsize=100)
def retrieve_context(query: str) -> str:
    retrieved_docs = db.similarity_search(query, k=TOP_K)
    return "\n\n".join([d.page_content for d in retrieved_docs])


# === Step 5: Interactive query loop ===
print("\n💬 Ready! Ask questions based on your file (type 'exit' to quit)\n")

while True:
    query = input("🔍 Your Question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break
    if not query:
        continue

    # Retrieve relevant chunks
    context = retrieve_context(query)

    # Build concise prompt
    prompt = (
        f"Use ONLY the following book content to answer clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer: \n"
        f"Answer in ONE CLEAR JAVA PROGRAMMING LANGUAGE SOLUTION without repeating yourself."
        f"DO NOT INCLUDE MULTIPLE ANSWERS."
        f"When ever requested for java code YOU SHOULD share only ONE JAVA CODE to satisfy the query."
    )

    # Generate streaming answer
    print("\n🧠 Answer:\n")
    stream = generator(
        prompt,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        repetition_penalty=1.2,  # discourages repeating
        return_full_text=False
    )

    # Stream tokens as they are generated
    for token in stream:
        print(token.get("generated_text", ""), end="", flush=True)
    print("\n" + "-" * 60 + "\n")
