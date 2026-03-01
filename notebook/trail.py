# ============================================================
# End-to-End RAG System with Voice & Text Input/Output
# ============================================================
# Features:
# - Audio Input: Record voice → ASR (Whisper) → RAG retrieval → LLM → TTS output
# - Text Input: Type question → RAG retrieval → LLM → Text output (optional TTS)
# - Multi-language: Supports English and Hindi
# - Vector Store: ChromaDB with legal-BERT embeddings
# ============================================================
# 0. Imports & Setup
# ============================================================

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from transformers import pipeline
import torch
from scipy.signal import resample

import gradio as gr
from dotenv import load_dotenv
from difflib import SequenceMatcher
import uuid
from openai import OpenAI
import tempfile
import requests
from bs4 import BeautifulSoup

# --- Load .env from project root ---
# Try multiple possible locations for .env file
script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
possible_env_paths = [
    script_dir / ".." / ".env",  # Project root (if script is in notebook/)
    script_dir / ".env",          # Same directory as script
    Path.cwd() / ".env",          # Current working directory
    Path.home() / ".env",         # Home directory (fallback)
]

env_loaded = False
for env_path in possible_env_paths:
    env_path = env_path.resolve()
    if env_path.exists():
        print(f"Found .env at: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
        env_loaded = True
        break
    else:
        print(f"Tried .env at: {env_path} (not found)")

if not env_loaded:
    # Try loading from current directory without specific path
    print("Attempting to load .env from current directory...")
    load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f" API key loaded (length: {len(api_key)} chars)")
else:
    print("WARNING: OPENAI_API_KEY not found in environment!")
    print("   Please create a .env file in the project root with:")
    print("   OPENAI_API_KEY=your_api_key_here")
    print("   Or set the environment variable directly.")

# 
# 1. PDF Loading (from ../data/pdf)
# 
def process_pdf(pdf_directory: str):
    """
    Recursively load all PDFs from a directory into LangChain Document objects.
    """
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir.resolve()}")

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
            all_documents.extend(documents)

            print(f"Loaded {len(documents)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"Failed to load {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


# Load all PDFs from data/pdf (relative to script location)
# Get the script's directory and resolve to absolute path
try:
    script_dir = Path(__file__).parent.resolve()
except NameError:
    script_dir = Path.cwd()
    
pdf_dir = script_dir.parent / "data" / "pdf"
print(f"Looking for PDFs in: {pdf_dir.resolve()}")
all_pdf_documents = process_pdf(str(pdf_dir.resolve()))

# ============================================================
# 2. Chunking with RecursiveCharacterTextSplitter
# ============================================================

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print("\nExample chunk:")
        print("Content:", split_docs[0].page_content[:200], "...")
        print("Metadata:", split_docs[0].metadata)
    return split_docs


chunks = split_documents(all_pdf_documents)

# ============================================================
# 3. Embedding Manager (legal-BERT)
# ============================================================

class EmbeddingManager:
    """Handles document embedding generation using sentence-transformers."""

    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load sentence-transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {dim}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            raise

    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded.")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("Embedding shape:", embeddings.shape)
        return embeddings

    def get_embedding_dimension(self) -> int:
        if self.model is None:
            raise RuntimeError("Embedding model not loaded.")
        return self.model.get_sentence_embedding_dimension()


embedding_manager = EmbeddingManager()

# ============================================================
# 4. VectorStore (ChromaDB) 
# ============================================================

class Vectorstore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "pdf_documents_legalbert",
        persist_directory: str = None,
        reset_collection: bool = True,   # <— always rebuild on script start
    ):
        # Set default persist directory relative to script location
        if persist_directory is None:
            try:
                script_dir = Path(__file__).parent.resolve()
            except NameError:
                script_dir = Path.cwd()
            persist_directory = str((script_dir.parent / "data" / "vector_store_legalbert").resolve())
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory).resolve())
        self.client = None
        self.collection = None
        self._initialize_store(reset_collection=reset_collection)

    def _initialize_store(self, reset_collection: bool):
        """Initialize the ChromaDB persistent store."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            print("Using persist directory:", self.persist_directory)

            self.client = chromadb.PersistentClient(path=self.persist_directory)

            if reset_collection:
                # Delete old collection if exists, so we always reflect ../data/pdf
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    print(f"No existing collection named {self.collection_name} to delete.")

            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={"description": "PDF document embeddings for RAG (legal-BERT)"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection_count()}")
        except Exception as e:
            print("Error initializing vector store:", e)
            raise

    def collection_count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    def add_documents(self, documents, embeddings: np.ndarray, batch_size: int = 1000):
        """
        Add documents and embeddings to ChromaDB in safe batches.

        Args:
            documents: list of LangChain Document objects
            embeddings: np.ndarray of shape (n_docs, emb_dim)
            batch_size: max documents per Chroma add() call
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        n_docs = len(documents)
        print(f"Adding {n_docs} documents to vector store (batch_size={batch_size})...")
        
        # Handle empty documents case
        if n_docs == 0:
            print("⚠️ No documents to add. Skipping vector store update.")
            return
        
        print("Embedding dim:", len(embeddings[0]))

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        # send to Chroma in batches (to avoid max-batch error)
        for start in range(0, n_docs, batch_size):
            end = min(start + batch_size, n_docs)
            print(f"  → Adding batch {start}:{end} ({end - start} docs)")
            try:
                self.collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents_text[start:end],
                )
            except Exception as e:
                print(f"Error adding batch {start}:{end} to vector store:", e)
                raise

        print(f"✅ Finished adding {n_docs} documents.")
        print(f"Total documents in collection: {self.collection.count()}")


# instantiate AND rebuild the vector store from ../data/pdf
vectorstore = Vectorstore(reset_collection=True)

print("Vector store is being rebuilt – computing embeddings and adding documents...")
if len(chunks) == 0:
    print("WARNING: No chunks found! Check if PDFs were loaded correctly.")
    print("PDF directory: ../data/pdf")
    print("   Make sure PDF files exist in that directory.")
else:
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embedding(texts)
    vectorstore.add_documents(chunks, embeddings, batch_size=1000)

# ============================================================
# 5. RAG Retriever (improved, invoice-aware)
# ============================================================

class RAGRetriever:
    """Improved retriever with fuzzy search + keyword/filename boost."""

    def __init__(self, vectorstore: Vectorstore, embedding_manager: EmbeddingManager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def _fuzzy_score(self, a: str, b: str) -> float:
        """Boost short queries using fuzzy matching on the first 300 chars."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _filename_boost(self, meta: Dict[str, Any], query: str) -> float:
        name = (meta.get("source_file") or "").lower()
        q = query.lower()
        score = 0.0

        # Strong boost for invoice / aws keywords in filename
        if "invoice" in name:
            score += 0.20
        if "aws" in name or "amazon" in name:
            score += 0.20

        # Light boost if query itself contains aws/invoice
        if "invoice" in q:
            score += 0.10
        if "aws" in q or "amazon" in q:
            score += 0.10

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"\n🔍 RAG retrieve() query: {query!r}, top_k={top_k}")

        # 1) Embed query
        query_emb = self.embedding_manager.generate_embedding([query])[0]

        # 2) Query Chroma
        try:
            results = self.vectorstore.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
            )
        except Exception as e:
            print("Error querying Chroma:", e)
            return []

        retrieved_docs: List[Dict[str, Any]] = []

        if not results or not results.get("documents") or not results["documents"][0]:
            print("No documents returned from Chroma.")
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        ids = results["ids"][0]

        for i, (doc_id, content, meta, dist) in enumerate(
            zip(ids, docs, metas, dists)
        ):
            # Chroma returns L2 distance, convert to similarity (0-1 range)
            # Normalize distance to similarity: smaller distance = higher similarity
            # Using exponential decay to convert distance to similarity
            cosine_like = 1.0 / (1.0 + dist) if dist > 0 else 1.0
            fuzzy = self._fuzzy_score(query, content[:300])

            kw_content = content.lower()
            query_lower = query.lower()
            # Expanded keywords based on query and content
            keywords = ["aws", "amazon", "invoice", "billing", "account", "h1b", "h-1b", "visa", "immigration", 
                       "petition", "uscis", "work", "employment", "sponsor", "status"]
            kw_hits = sum(1 for k in keywords if k in kw_content)
            # Also check if query keywords match
            query_keywords = ["h1b", "h-1b", "visa", "invoice", "aws", "amazon"]
            query_kw_match = sum(1 for k in query_keywords if k in query_lower)
            keyword_boost = (kw_hits * 0.05) + (query_kw_match * 0.03)  # text keyword boost

            filename_boost = self._filename_boost(meta, query)

            final_score = (
                0.55 * cosine_like
                + 0.20 * fuzzy
                + 0.10 * keyword_boost
                + 0.15 * filename_boost
            )

            print(
                f"  → cand {i+1}: id={doc_id}, "
                f"cos={cosine_like:.3f}, fuzzy={fuzzy:.3f}, "
                f"txt_kw={keyword_boost:.3f}, file_boost={filename_boost:.3f}, "
                f"final={final_score:.3f}, source={meta.get('source_file')}"
            )

            retrieved_docs.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": meta,
                    "similarity_score": final_score,
                    "distance": dist,
                    "rank": i + 1,
                }
            )

        # sort by final_score descending and keep top_k
        retrieved_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
        print("\nTop reranked docs:")
        for d in retrieved_docs[:top_k]:
            print(
                f" score={d['similarity_score']:.3f}, "
                f"file={d['metadata'].get('source_file')}, len={d['metadata'].get('content_length')}"
            )
        return retrieved_docs[:top_k]


rag_retriever = RAGRetriever(vectorstore, embedding_manager)

# Quick sanity test (text-only RAG, no voice yet)
print("\n" + "="*60)
print("Testing RAG retrieval...")
print("="*60)
test_results = rag_retriever.retrieve("aws invoice", top_k=3)
print(f"\n✅ RAG Test Results: Retrieved {len(test_results)} documents")
if test_results:
    for i, d in enumerate(test_results, 1):
        print(f"  {i}. Score: {d['similarity_score']:.3f}, File: {d['metadata'].get('source_file')}")
else:
    print("  ⚠️ WARNING: No documents retrieved! Check if vector store has documents.")
    print(f"  Vector store count: {vectorstore.collection_count()}")
print("="*60 + "\n")

# ============================================================
# 6. Initialize Chat Model (OpenAI via LangChain)
# ============================================================

# Verify API key is set before initializing model
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found! Please set it in your .env file or environment variables.\n"
        "Create a .env file in the project root with: OPENAI_API_KEY=your_key_here"
    )

model = init_chat_model("gpt-4o", temperature=0.1, max_tokens=1024)
print("Chat model initialized successfully")

# ============================================================
# 7. Advanced RAG helper with fallback
# ============================================================

H1B_UPDATES_URL = "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations"


def fetch_h1b_updates() -> str:
    """
    Fetch and lightly parse the official USCIS H‑1B Specialty Occupations page.

    We focus on:
    - The main ALERT about recent changes
    - The 'Last Reviewed/Updated' date
    - Key sections about cap, portability, and eligibility

    Reference: USCIS H‑1B Specialty Occupations page [USCIS H‑1B](https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations)
    """
    try:
        print(f"🌐 Fetching latest H‑1B updates from USCIS: {H1B_UPDATES_URL}")
        resp = requests.get(H1B_UPDATES_URL, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f" Failed to fetch USCIS H‑1B page: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Get alert text (the bold 'ALERT:' section near the top)
    alert_text = ""
    for strong in soup.find_all("strong"):
        text = strong.get_text(strip=True)
        if "ALERT" in text.upper():
            parent = strong.parent
            alert_text = parent.get_text(" ", strip=True)
            break

    # Get last updated date
    page_text = soup.get_text("\n", strip=True)
    last_updated = ""
    marker = "Last Reviewed/Updated:"
    if marker in page_text:
        idx = page_text.index(marker) + len(marker)
        last_updated = page_text[idx : idx + 40].split("\n")[0].strip()

    # Extract main H‑1B content around the heading
    main_section = ""
    h1 = soup.find("h1")
    if h1:
        # Grab following paragraphs and headings
        parts = []
        for sib in h1.find_all_next():
            if sib.name in {"h2", "h3"} and "More Information" in sib.get_text():
                break
            if sib.name in {"p", "h2", "h3", "ul", "ol"}:
                parts.append(sib.get_text(" ", strip=True))
            if len(" ".join(parts)) > 4000:
                break
        main_section = "\n\n".join(parts)

    context_pieces = []
    if last_updated:
        context_pieces.append(f"Last Reviewed/Updated: {last_updated}")
    if alert_text:
        context_pieces.append(f"USCIS ALERT: {alert_text}")
    if main_section:
        context_pieces.append("Key H‑1B information from USCIS:\n" + main_section)

    context = "\n\n".join(context_pieces).strip()
    print(f" H‑1B context length from USCIS: {len(context)} characters")
    return context


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm,
    top_k: int = 5,
    return_context: bool = False,
) -> Dict[str, Any]:
    """
    RAG pipeline:
    - If query is about H‑1B → fetch live updates from USCIS and answer from that context.
    - Otherwise → use PDF RAG.
    """

    q_lower = query.lower()
    h1b_keywords = ["h1b", "h-1b", "h1-b", "h 1b", "h 1-b"]

    # 1) Special handling for H‑1B queries: use official USCIS page
    if any(k in q_lower for k in h1b_keywords):
        print("🛈 H‑1B keywords detected – using USCIS H‑1B page instead of PDF RAG.")
        h1b_context = fetch_h1b_updates()

        if not h1b_context:
            print("⚠️ Could not fetch H‑1B context – falling back to normal RAG.")
        else:
            prompt = f"""
You are an expert immigration assistant. Use ONLY the following official USCIS information
about H‑1B Specialty Occupations to answer the question. If the question asks for recent
updates or changes, focus on the ALERT and any time‑sensitive information.

CONTEXT (from USCIS H‑1B page):
{h1b_context}

QUESTION:
{query}

Provide a clear, accurate answer. If you summarize or interpret, keep it faithful to the USCIS source.
Mention that the information is sourced from the official USCIS H‑1B Specialty Occupations page.
""".strip()

            print("\n Sending H‑1B question to LLM with USCIS context...")
            resp = llm.invoke([HumanMessage(content=prompt)])
            result = {
                "answer": resp.content,
                "source": [
                    {
                        "id": "uscis_h1b_page",
                        "content": h1b_context,
                        "metadata": {
                            "source_file": "USCIS H-1B Specialty Occupations page",
                            "url": H1B_UPDATES_URL,
                            "type": "web",
                        },
                        "similarity_score": 1.0,
                        "distance": 0.0,
                        "rank": 1,
                    }
                ],
                "confidence": 1.0,
            }
            if return_context:
                result["context"] = h1b_context
            return result

    # 2) Normal PDF RAG path
    # Increase top_k to get more relevant documents for better context
    retrieved = retriever.retrieve(query, top_k=max(top_k, 10))

    # Fallback: no docs → answer directly with GPT (no RAG)
    if not retrieved:
        print(" WARNING: No documents retrieved from RAG!")
        print("   This means the vector store might be empty or the query doesn't match any documents.")
        print("   Answering from LLM knowledge only (not using RAG context).")
        direct_prompt = (
            "You are a helpful assistant. Answer the following question clearly:\n\n"
            f"{query}\n\nAnswer:"
        )
        resp = llm.invoke([HumanMessage(content=direct_prompt)])
        return {
            "answer": resp.content,
            "source": [],
            "confidence": 0.0,
            "context": "" if return_context else None,
        }

    # Normal RAG path - USE THE RETRIEVED CONTEXT
    # Limit context length to avoid token limits, but include top documents
    max_context_length = 3000  # characters
    context_parts = []
    current_length = 0

    for d in retrieved:
        doc_text = f"[Source: {d['metadata'].get('source_file', 'unknown')}]\n{d['content']}"
        if current_length + len(doc_text) > max_context_length and context_parts:
            break
        context_parts.append(doc_text)
        current_length += len(doc_text)

    context = "\n\n".join(context_parts)

    print(f"\n📄 Using {len(context_parts)}/{len(retrieved)} retrieved documents for RAG context")
    print(f"Context length: {len(context)} characters")
    print(f"Top document similarity: {retrieved[0]['similarity_score']:.3f}")
    print(f"Top document source: {retrieved[0]['metadata'].get('source_file', 'unknown')}")
    print(f"Top document preview: {retrieved[0]['content'][:200]}...")

    prompt = f"""You are a helpful assistant answering questions based on PDF documents. Use the context below to answer the question.

IMPORTANT: 
- If the context contains ANY relevant information (even partial), use it to answer
- Look for keywords, concepts, or related information in the context
- Be flexible in matching the question to the context - the question might be phrased differently
- Only say "I cannot find this information" if the context is completely unrelated to the question

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {query}

Answer based on the context above. If relevant information exists, use it. If not, provide a helpful answer from your knowledge:""".strip()

    print("\n Sending to LLM with RAG context...")
    resp = llm.invoke([HumanMessage(content=prompt)])

    result = {
        "answer": resp.content,
        "source": retrieved,
        "confidence": max(d["similarity_score"] for d in retrieved),
    }
    if return_context:
        result["context"] = context

    return result

# ============================================================
# 8. ASR (Whisper) + Language Utilities
# ============================================================

# 1) ASR – Whisper small (multilingual) → force CPU for stability
asr_device = "cpu"
print("ASR device:", asr_device)

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=-1,  # CPU
)

# 2) Simple translations via LLM

def translate_en_to_hi(text: str) -> str:
    prompt = f"Translate the following English text to Hindi:\n\n{text}"
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def translate_hi_to_en(text: str) -> str:
    prompt = f"Translate the following Hindi text to English:\n\n{text}"
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# 3) Simple script detector

def detect_language_script(text: str) -> str:
    """
    Very simple detection:
    - if Devanagari characters present -> 'hi'
    - otherwise -> 'en'
    """
    for ch in text:
        if "\u0900" <= ch <= "\u097F":  # Devanagari Unicode range
            return "hi"
    return "en"

# 4) Transcription wrapper with manual resampling
#    lang_hint: "en", "hi", or "auto" from dropdown

def transcribe_audio_to_text(audio_array: np.ndarray, sr: int, lang_hint: str = "auto") -> str:
    """
    Use Whisper to transcribe audio (Hindi or English).
    Manually resample to 16 kHz to avoid torchaudio dependency.
    """
    if audio_array is None or len(audio_array) == 0:
        print("transcribe_audio_to_text: empty audio array")
        return ""

    print("transcribe_audio_to_text: input sr:", sr, "shape:", audio_array.shape)

    target_sr = 16000

    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    # Resample if needed
    if sr != target_sr:
        duration = audio_array.shape[0] / sr
        new_length = int(duration * target_sr)
        audio_array = resample(audio_array, new_length)
        sr = target_sr

    audio_array = audio_array.astype("float32")

    asr_inputs = {"array": audio_array, "sampling_rate": sr}

    # Make lang_hint a soft hint, but don't break if "auto"
    if lang_hint == "en":
        print("ASR hint: English")
        out = asr(asr_inputs, generate_kwargs={"language": "en"})
    elif lang_hint == "hi":
        print("ASR hint: Hindi")
        out = asr(asr_inputs, generate_kwargs={"language": "hi"})
    else:
        print("ASR hint: auto-detect")
        out = asr(asr_inputs)

    text = out.get("text", "").strip()
    print("ASR text:", text if text else "(empty)")
    return text

# ============================================================
# 9. TTS using OpenAI TTS API
# ============================================================

# Initialize OpenAI client for TTS
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(" OpenAI TTS client initialized")
else:
    print(" OpenAI API key not found - TTS will be disabled")

def synthesize_tts_gpt(text: str, output_path: str = None, voice: str = "alloy") -> str:
    """
    Generate speech from text using OpenAI TTS API.
    
    Args:
        text: Text to convert to speech
        output_path: Optional path to save audio file. If None, uses temp file.
        voice: Voice to use ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
    
    Returns:
        Path to generated audio file, or None if TTS fails
    """
    if not openai_client:
        print(" TTS unavailable: OpenAI API key not set")
        return None
    
    if not text or not text.strip():
        print(" TTS: Empty text provided")
        return None
    
    try:
        # Use temp file with .mp3 extension for Gradio compatibility
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=tempfile.gettempdir())
            output_path = temp_file.name
            temp_file.close()
        
        # Truncate text to OpenAI TTS limit (4096 chars)
        text_to_speak = text[:4000] if len(text) > 4000 else text
        if len(text) > 4000:
            print(f"Text truncated from {len(text)} to 4000 chars for TTS")
        
        print(f"🎤 Generating TTS audio (text length: {len(text_to_speak)} chars, voice: {voice})...")
        
        # Call OpenAI TTS API
        response = openai_client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality (more expensive)
            voice=voice,
            input=text_to_speak
        )
        
        # Save audio to file
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        # Verify file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f" TTS audio saved successfully: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        else:
            print(f" TTS file was not created properly")
            return None
        
    except Exception as e:
        print(f"TTS error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# 10. RAG + Language Handling (text IO)
# ============================================================

def rag_answer_with_lang(
    user_text: str,
    retriever: RAGRetriever,
    llm_model,
    input_lang: str = "auto",   # 'auto', 'en', 'hi'
    output_lang: str = "en",    # 'en', 'hi'
    do_tts: bool = True,
) -> Dict[str, Any]:
    text = user_text.strip()
    if not text:
        return {
            "query_detected_lang": None,
            "query_used_for_rag": "",
            "answer_en": "",
            "answer_final": "",
            "audio_path": None,
            "rag_sources": [],
        }

    # 1) Detect language
    if input_lang == "auto":
        detected = detect_language_script(text)
    else:
        detected = input_lang

    print("Detected input language:", detected)

    # 2) Translate Hindi query -> English for RAG
    if detected == "hi":
        query_en = translate_hi_to_en(text)
        print("HI → EN query:", query_en)
    else:
        query_en = text

    # 3) RAG
    rag_result = rag_advanced(
        query_en,
        retriever,
        llm_model,
        top_k=10,  # Increased for better retrieval
        return_context=True,
    )

    answer_en = rag_result["answer"]
    sources = rag_result["source"]

    # 4) Output language + TTS based on selected output language
    audio_path = None
    if output_lang == "hi":
        answer_final = translate_en_to_hi(answer_en)
        print("EN → HI answer:", answer_final)
    else:
        answer_final = answer_en
    
    # Generate TTS ONLY if do_tts is True and we have text
    if do_tts and answer_final:
        print(f" Generating TTS audio for {output_lang} output...")
        # Use appropriate voice based on language
        voice = "nova" if output_lang == "hi" else "alloy"
        audio_path = synthesize_tts_gpt(answer_final, voice=voice)
        if audio_path:
            print(f"TTS audio generated: {audio_path}")
        else:
            print("TTS generation failed")

    return {
        "query_detected_lang": detected,
        "query_used_for_rag": query_en,
        "answer_en": answer_en,
        "answer_final": answer_final,
        "audio_path": audio_path,
        "rag_sources": sources,
    }

# ============================================================
# 11. Gradio App: Voice + Text Chatbot UI
# ============================================================

def gradio_rag_voice(audio, input_lang_choice, output_lang_choice, history):
    """
    audio: (sr, np.ndarray) from gr.Audio(type='numpy')
    history: chat history for Chatbot (list of messages with role and content)
    """
    if history is None:
        history = []

    if audio is None:
        debug = (
            "Audio is None → frontend didn't send anything.\n"
            "Check:\n"
            "1) Browser mic permission is ALLOWED for 127.0.0.1\n"
            "2) You see a waveform in the Gradio audio component before clicking the button\n"
            "3) Try Chrome instead of Safari/Firefox\n"
        )
        history.append({"role": "user", "content": " No audio detected"})
        history.append({"role": "assistant", "content": "Please record again."})
        return history, None, debug

    # Handle different audio input formats from Gradio
    try:
        if isinstance(audio, tuple):
            sr, wav = audio
        elif isinstance(audio, dict):
            sr = audio.get("sampling_rate", 16000)
            wav = audio.get("array", None)
        else:
            # Try to extract directly
            sr = getattr(audio, "sampling_rate", 16000)
            wav = getattr(audio, "array", audio)
    except Exception as e:
        debug = f" Could not unpack audio: {e}, type={type(audio)}, raw={audio}"
        history.append({"role": "user", "content": " Audio error"})
        history.append({"role": "assistant", "content": "Please try recording again."})
        return history, None, debug

    print(f"Received audio from Gradio. sr: {sr}, type: {type(wav)}, shape: {getattr(wav, 'shape', None)}")

    if wav is None:
        debug = "Audio array is None."
        history.append({"role": "user", "content": "🎤 (silence?)"})
        history.append({"role": "assistant", "content": "No speech detected, please try again."})
        return history, None, debug
    
    # Convert to numpy array if needed
    if not isinstance(wav, np.ndarray):
        try:
            wav = np.array(wav)
        except:
            debug = f"Could not convert audio to numpy array: {type(wav)}"
            history.append({"role": "user", "content": " Audio error"})
            history.append({"role": "assistant", "content": "Please try recording again."})
            return history, None, debug

    if len(wav) == 0:
        debug = " Empty audio array received from Gradio."
        history.append({"role": "user", "content": "🎤 (silence?)"})
        history.append({"role": "assistant", "content": "No speech detected, please try again."})
        return history, None, debug

    # Ensure mono and correct dtype
    if wav.ndim > 1:
        wav = wav[:, 0] if wav.shape[1] > 0 else wav.flatten()

    wav = wav.astype("float32")
    
    # Normalize audio
    if wav.max() > 1.0:
        wav = wav / (wav.max() + 1e-8)

    # Step 1: ASR (pass dropdown hint)
    print(f"\n Processing audio: sr={sr}, shape={wav.shape}, duration={len(wav)/sr:.2f}s")
    spoken_text = transcribe_audio_to_text(wav, sr, lang_hint=input_lang_choice)
    if not spoken_text:
        debug = "ASR returned empty text – maybe audio is too quiet or too short."
        history.append({"role": "user", "content": "🎤 (silence?)"})
        history.append({"role": "assistant", "content": "No speech detected, please try again."})
        return history, None, debug
    
    print(f" ASR transcribed: '{spoken_text}'")

    # Step 2: RAG + language pipeline
    print(f"\n Starting RAG pipeline for query: '{spoken_text}'")
    try:
        result = rag_answer_with_lang(
            spoken_text,
            rag_retriever,
            model,
            input_lang=input_lang_choice,
            output_lang=output_lang_choice,
            do_tts=True,   # Generate TTS audio output
        )
        print(f"RAG pipeline completed. Answer length: {len(result.get('answer_final', ''))}")
    except Exception as e:
        print(f" Error in RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        debug = f"Error processing query: {str(e)}"
        history.append({"role": "user", "content": spoken_text})
        history.append({"role": "assistant", "content": f"Sorry, an error occurred: {str(e)}"})
        return history, None, debug

    final_answer = result["answer_final"]
    audio_path = result["audio_path"]
    detected_lang = result["query_detected_lang"]

    # Build debug info
    sources_list = [s['metadata'].get('source_file', 'unknown') for s in result['rag_sources']]
    debug_text = (
        f"✅ Detected language: {detected_lang}\n"
        f"🎤 ASR heard: {spoken_text}\n"
        f"🔍 Query sent to RAG: {result['query_used_for_rag']}\n"
        f"📄 RAG sources ({len(result['rag_sources'])}): {sources_list}\n"
        f"🎵 TTS audio: {'Generated' if audio_path else 'Not generated'}\n"
        f"🌐 Output language: {output_lang_choice}"
    )

    # Update chat history (messages format)
    history.append({"role": "user", "content": spoken_text})
    history.append({"role": "assistant", "content": final_answer})

    return history, audio_path, debug_text


def gradio_rag_text(user_text, input_lang_choice, output_lang_choice, history):
    """
    Text fallback: lets you test RAG + LLM without audio.
    """
    if history is None:
        history = []

    text = (user_text or "").strip()
    if not text:
        history.append({"role": "user", "content": "Empty message"})
        history.append({"role": "assistant", "content": "Please type a question."})
        return history, "", "No text provided."

    result = rag_answer_with_lang(
        text,
        rag_retriever,
        model,
        input_lang=input_lang_choice,
        output_lang=output_lang_choice,
        do_tts=True,  # Enable TTS for text input too, based on output_lang_choice
    )

    final_answer = result["answer_final"]
    detected_lang = result["query_detected_lang"]
    audio_path = result["audio_path"]
    sources_list = [s['metadata'].get('source_file', 'unknown') for s in result['rag_sources']]

    debug_text = (
        f"[TEXT MODE]\n"
        f"✅ Detected language: {detected_lang}\n"
        f"🔍 Query sent to RAG: {result['query_used_for_rag']}\n"
        f"📄 RAG sources ({len(result['rag_sources'])}): {sources_list}\n"
        f"🎵 TTS audio: {'Generated' if audio_path else 'Not generated'}\n"
        f"🌐 Output language: {output_lang_choice}"
    )

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": final_answer})

    # Return audio path if generated, clear textbox
    return history, audio_path if audio_path else "", debug_text


with gr.Blocks() as demo:
    gr.Markdown("## LexiTalks")

    with gr.Row():
        input_lang = gr.Dropdown(
            ["auto", "en", "hi"],
            value="auto",
            label="Input language detection (for ASR + RAG)",
            info="For English questions like 'AWS invoice', choose 'en' for best results.",
        )
        output_lang = gr.Dropdown(
            ["en", "hi"],
            value="en",
            label="Answer language",
            info="Choose how you want the RAG answer.",
        )

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=False,
            interactive=True,
            label="🎤 Speak your question (click mic, talk, then stop)",
        )
        text_in = gr.Textbox(
            label="💬 Or type your question",
            placeholder="e.g., Show me the AWS invoice details",
        )

    with gr.Row():
        ask_audio_btn = gr.Button("Ask with Voice")
        ask_text_btn = gr.Button("Ask with Text")

    chatbot = gr.Chatbot(
        label="Chat History",
        type="messages",  # Use new messages format
    )
    audio_out = gr.Audio(
        label="Audio Answer (TTS output - English or Hindi)",
        type="filepath",
    )
    debug_box = gr.Textbox(
        label="Debug Info",
        lines=8,
    )

    # Voice flow
    ask_audio_btn.click(
        fn=gradio_rag_voice,
        inputs=[audio_in, input_lang, output_lang, chatbot],
        outputs=[chatbot, audio_out, debug_box],
    )

    # Text flow (with optional audio output)
    ask_text_btn.click(
        fn=gradio_rag_text,
        inputs=[text_in, input_lang, output_lang, chatbot],
        outputs=[chatbot, audio_out, debug_box],
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gradio interface...")
    print("="*60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )
