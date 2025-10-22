# OfflineLLMBookAssistant
An offline AI learning assistant that helps students study anywhere, even without internet access.

# 🧠 Local SLM Q&A Engine
A lightweight offline question–answering system powered by a Small Language Model (Phi-1.5) and FAISS semantic search.  
Reads your document once and lets you ask natural questions anytime — fully offline, GPU-optimized for 4 GB VRAM.

### Features
- ⚙️ Offline inference (no API keys)
- 🧩 Semantic search with FAISS
- 🧠 Quantized Phi model (4-bit)
- 💬 Interactive Q&A loop
- 💾 Persistent local index


## steps
- Use PDFtoTXT.ipynb to convert PDF to text (You can use any book to do this)
- Use slm_query_engine_fast_book2.py to use the above text as a training material to the SLM (You might need to change the file name in file to consume other text files)
- run `python slm_query_engine_fast_book2.py` for interactive chat bot that can answer your queries from book
