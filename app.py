# app.py — RAG-Based BHB Chatbot (FAISS on Google Drive)
import os
import zipfile
from typing import List

import streamlit as st
import numpy as np

# Vector store + LLM
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Utilities
import gdown
import re
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Page setup & header
# -----------------------------
st.set_page_config(page_title="RAG-Based BHB Chatbot", page_icon="🔎", layout="wide")
st.title("Beta-hydroxybutyrate Chatbot")
st.caption(
    "This is an LLM model trained exclusively with scientific literature about BHB. "
    "It should give you a more precise and in-depth answer about BHB than standard LLMs like ChatGPT. "
    "It is also programmed to give references for all of its claims. "
)

# -----------------------------
# Configuration via st.secrets
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
DRIVE_FILE_ID = st.secrets.get("DRIVE_FILE_ID", None)
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
INDEX_SUBDIR = st.secrets.get("INDEX_SUBDIR", "faiss_index")
CUSTOM_SYSTEM_PROMPT = st.secrets.get("SYSTEM_PROMPT", "").strip()

if not OPENAI_API_KEY or not DRIVE_FILE_ID or not EMBEDDING_MODEL:
    st.error("Missing required secrets. Please set OPENAI_API_KEY, DRIVE_FILE_ID, and EMBEDDING_MODEL in Secrets.")
    st.stop()

#Examples

EXAMPLE_PROMPTS = [
    "Summarize the key mechanisms by which BHB affects mitochondrial function.",
    "How does BHB influence inflammatory pathways in human immune cells?",
    "Explain what is known about BHB and its role in neuroprotection.",

]

# -----------------------------
# PubMed helpers
# -----------------------------
def extract_pmid_from_content(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"^---\s*PUBMED\s+ABSTRACT\s*\(\s*(?:PMID[: ]*)?(\d{5,9})\s*\)", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1)
    m = re.search(r"\bPMID\s*[:#]?\s*(\d{5,9})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\((\d{5,9})\)", text[:200])
    return m.group(1) if m else None

def extract_pmid(doc) -> str | None:
    md = (getattr(doc, "metadata", None) or {})
    for k in ("pmid", "PMID", "source_article_id", "id"):
        v = md.get(k, None)
        if v and str(v).isdigit():
            return str(v)
    # If not in metadata, try content
    content = getattr(doc, "page_content", "") or ""
    return extract_pmid_from_content(content)

def fetch_pubmed_title(pmid: str) -> str | None:
    if not pmid:
        return None
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        h1 = soup.find("h1", class_="heading-title")
        if h1:
            title_text = " ".join(h1.get_text(strip=True).split())
            return title_text
    except Exception:
        return None
    return None

def get_doc_title_or_snippet(doc) -> str:
    md = (getattr(doc, "metadata", None) or {})
    for k in ("title", "paper_title", "article_title"):
        v = md.get(k)
        if v:
            return str(v)
    # Fallback: PubMed title if we have pmid
    pmid = extract_pmid(doc)
    if pmid:
        t = fetch_pubmed_title(pmid)
        if t:
            return t
    # Finally, first non-empty line of content
    content = (getattr(doc, "page_content", "") or "")
    for ln in content.splitlines():
        ln = ln.strip()
        if ln and not ln.upper().startswith("--- PUBMED ABSTRACT"):
            return ln if len(ln) <= 150 else ln[:150] + "…"
    return "(no title)"

# -----------------------------
# Helpers
# -----------------------------
def _download_from_drive(file_id_or_url: str, dest_path: str) -> str:
    """
    Download the FAISS index ZIP file.

    `file_id_or_url` can be either:
      • a Google Drive *file ID* (e.g. '1AbCdEf...')
      • or a full download URL (https://...).

    In your `secrets.toml`, set `DRIVE_FILE_ID` to either form.
    """
    # If this looks like a full URL, use it directly; otherwise assume it is a Drive file ID.
    if file_id_or_url.startswith("http://") or file_id_or_url.startswith("https://"):
        url = file_id_or_url
    else:
        url = f"https://drive.google.com/uc?id={file_id_or_url}"

    with st.spinner("Downloading FAISS index..."):
        out = gdown.download(url, dest_path, quiet=False)
    if out is None:
        raise RuntimeError(
            "Failed to download FAISS index. Check the sharing permissions and that the URL / file ID is correct."
        )
    return out

def _ensure_index_ready(cache_dir: str = ".cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local_zip = os.path.join(cache_dir, "faiss_index.zip")
    local_index_dir = os.path.join(cache_dir, INDEX_SUBDIR)
    index_files = ["index.faiss", "index.pkl"]
    # If index already extracted, reuse
    if os.path.isdir(local_index_dir) and all(os.path.exists(os.path.join(local_index_dir, f)) for f in index_files):
        return local_index_dir
    # Otherwise download and extract
    _download_from_drive(DRIVE_FILE_ID, local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(cache_dir)
    # If extracted files are not in the subdir, move them
    if not os.path.isdir(local_index_dir):
        os.makedirs(local_index_dir, exist_ok=True)
        for member in os.listdir(cache_dir):
            if member.endswith(".faiss") or member.endswith(".pkl") or member.endswith(".json"):
                os.replace(os.path.join(cache_dir, member), os.path.join(local_index_dir, member))
    # Check again
    if not all(os.path.exists(os.path.join(local_index_dir, f)) for f in index_files):
        raise FileNotFoundError("Extracted index folder is missing required files (index.faiss and index.pkl).")
    return local_index_dir

def docs_to_prompt_context(docs: List[Document], max_chars: int = 6000) -> str:
    """
    Turn retrieved docs into a single context string with soft character budget.
    """
    chunks = []
    total = 0
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        if not text.strip():
            continue
        # Optional metadata heading
        pmid = extract_pmid(d)
        title = get_doc_title_or_snippet(d)
        heading_parts = []
        if title and title != "(no title)":
            heading_parts.append(title)
        if pmid:
            heading_parts.append(f"PMID {pmid}")
        heading = " — ".join(heading_parts) if heading_parts else None

        if heading:
            text_block = f"### {heading}\n{text}"
        else:
            text_block = text

        block_len = len(text_block)
        if total + block_len > max_chars:
            # If nothing yet, at least include one doc
            if not chunks:
                chunks.append(text_block[:max_chars])
            break
        chunks.append(text_block)
        total += block_len

    if not chunks:
        return "(no relevant context found in the index.)"
    return "\n\n---\n\n".join(chunks)

# -----------------------------
# Load heavy resources (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading embeddings, vector store, and LLM...")
def load_resources():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    # 1. Ensure FAISS index is ready
    index_dir = _ensure_index_ready()
    # 2. Embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    # 3. Vectorstore
    vectorstore = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    # 4. Sanity check on embedding dims
    index_dim = vectorstore.index.d
    test_vec = embeddings.embed_query("ping")
    query_dim = len(test_vec)
    if query_dim != index_dim:
        raise ValueError(
            f"Embedding dimension mismatch: index dim = {index_dim}, embedding model dim = {query_dim}. "
            "Make sure EMBEDDING_MODEL matches the model used to build the FAISS index."
        )
    # 5. LLM
    if CUSTOM_SYSTEM_PROMPT:
        SYSTEM_PROMPT = CUSTOM_SYSTEM_PROMPT
    else:
        SYSTEM_PROMPT = (
            "You are an expert assistant on beta-hydroxybutyrate (BHB). You answer strictly based on the provided "
            "context, which comes from scientific literature. If the context does not contain enough information to "
            "answer confidently, you state this explicitly.\n\n"
            "When you make factual statements, you cite supporting PubMed IDs (PMIDs) from the context. "
            "If you cannot find a specific PMID for a claim, be transparent that it is uncertain or speculative.\n\n"
            "Always:\n"
            "1) Start with a concise, direct answer (2–4 sentences).\n"
            "2) Then provide a more detailed explanation.\n"
            "3) Finally, list key PMIDs with 1–2 bullet points each summarizing their relevance.\n"
            "If you mention numerical results, specify population, dose, and key conditions whenever possible."
        )
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nGive a direct answer first, then cite PMIDs."),
        ]
    )
    doc_chain = (
        {
            "context": RunnableLambda(lambda x: docs_to_prompt_context(x.get("context", []))),
            "question": RunnableLambda(lambda x: x.get("question", "")),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "doc_chain": doc_chain,
    }

# -----------------------------
# Main UI
# -----------------------------
resources = load_resources()
retriever = resources["retriever"]
doc_chain = resources["doc_chain"]

# Left: input; Right: retrieved context
col_input, col_context = st.columns([0.55, 0.45])

with col_input:
    st.subheader("Ask a question about BHB")
    user_question = st.text_area(
        "Question",
        value="Summarize key mechanisms by which BHB affects mitochondrial function.",
        height=100,
    )

    show_advanced = st.checkbox("Show advanced settings", value=False)
    if show_advanced:
        with st.expander("Retrieval & model settings", expanded=False):
            k = st.slider("Number of documents to retrieve (k)", min_value=3, max_value=20, value=8, step=1)
            retriever.search_kwargs["k"] = k

    st.markdown("**Or pick an example question:**")
    for i, ex in enumerate(EXAMPLE_PROMPTS, start=1):
        if st.button(ex, key=f"ex_{i}"):
            user_question = ex
            st.experimental_rerun()

    run_button = st.button("Run")

with col_context:
    st.subheader("Retrieved literature context")

    if "last_docs" not in st.session_state:
        st.session_state["last_docs"] = []

    last_docs = st.session_state["last_docs"]
    if last_docs:
        st.write("Below are the documents retrieved for the last answer:")
        for i, d in enumerate(last_docs, start=1):
            md = (getattr(d, "metadata", None) or {})
            pmid = extract_pmid(d)
            title = get_doc_title_or_snippet(d)
            url = None
            if pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            pmid_str = pmid if pmid else "N/A"
            left, right = st.columns([0.80, 0.20])
            with left:
                header = f"{i}. {title} — PMID {pmid_str}"
                with st.expander(header):
                    st.write(d.page_content)
            with right:
                if url:
                    st.link_button("View on PubMed", url, use_container_width=True)
                else:
                    st.caption("No PubMed link")
            st.divider()

# -----------------------------
# Run query
# -----------------------------
if run_button and user_question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        # Support both old-style retrievers (with get_relevant_documents)
        # and new Runnable-style retrievers (use .invoke).
        try:
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(user_question)
            else:
                # Newer LangChain retrievers behave like Runnables
                docs = retriever.invoke(user_question)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            docs = []

        st.session_state["last_docs"] = docs
        answer = doc_chain.invoke({"question": user_question, "context": docs})

    st.subheader("Answer")
    st.write(answer)
    if not docs:
        st.info("No documents were retrieved for this question. The answer may be based on prior model knowledge.")
elif not run_button:
    st.info("Enter a question and click **Run**, or choose an example below.")

