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
    "Describe the evidence for BHB’s impact on cardiovascular health markers.",
    "Compare endogenous BHB (e.g., from fasting) vs exogenous ketone supplements in human studies.",
    "Summarize how BHB interacts with histone deacetylases (HDACs) and gene expression.",
    "Describe the role of BHB in oxidative stress and redox homeostasis.",
    "What is known about BHB’s effects on insulin sensitivity and glucose metabolism?",
    "Explain how BHB may modulate the NLRP3 inflammasome and inflammatory cytokines.",
    "Detail the association between BHB and cognitive performance or brain energy metabolism.",
    "Summarize safety and adverse effects reported for exogenous ketone supplementation.",
    "Describe the dose–response relationships observed for BHB in human clinical trials.",
    "Explain what is known about BHB’s role in skeletal muscle function and exercise performance.",
    "Describe the evidence for BHB as a signaling metabolite beyond its role as an energy substrate.",
    "Summarize how BHB levels change in response to diet, fasting, and exercise.",
    "Explain how BHB may influence epigenetic regulation via histone modifications.",
    "Describe the pharmacokinetics of exogenous BHB (ketone esters/salts) in humans.",
    "Summarize the key differences between β-hydroxybutyrate and other ketone bodies.",
    "What is known about BHB’s effects on appetite and satiety regulation?",
    "Describe how BHB interacts with GPR109A (HCAR2) and downstream signaling pathways.",
    "Summarize how BHB might influence aging-related pathways and longevity.",
    "Explain how BHB may impact endothelial function and vascular health.",
    "Describe the relationship between BHB and markers of metabolic health (e.g., HbA1c, lipids).",
    "Summarize the evidence for BHB’s effects in neurodegenerative diseases (e.g., Alzheimer’s, Parkinson’s).",
    "How does BHB affect mitochondrial biogenesis and PGC-1α signaling?",
    "Describe what is known about BHB in the context of heart failure and cardiac metabolism.",
    "Summarize the interactions between BHB and autophagy pathways.",
    "Explain the role of BHB in the context of sepsis and critical illness.",
    "Describe the evidence for BHB’s effects on mood and psychiatric conditions.",
    "Summarize how BHB influences oxidative phosphorylation and ATP production.",
    "Explain how BHB impacts the balance between glycolysis and fatty acid oxidation.",
    "Describe BHB’s effects on the gut microbiome, if any have been reported.",
    "Summarize the role of BHB in regulating blood–brain barrier integrity.",
    "How does BHB influence lactate metabolism and MCT transporters?",
    "Describe BHB’s role in the context of cancer metabolism and potential therapeutic effects.",
    "Summarize the evidence for BHB’s effects on bone metabolism and osteoporosis risk.",
    "Explain how BHB may affect liver fat content and non-alcoholic fatty liver disease (NAFLD).",
    "Describe the relationship between BHB and stress hormone regulation (e.g., cortisol).",
    "Summarize how BHB interacts with sirtuins and NAD+ metabolism.",
    "Explain what is known about BHB’s effects on renal function and acid–base balance.",
    "Describe the time course of BHB changes after ingestion of ketone esters.",
    "Summarize how BHB levels correlate with performance outcomes in endurance athletes.",
    "Explain BHB’s role in hypoglycemia prevention or mitigation.",
    "Describe how BHB interacts with insulin signaling pathways in peripheral tissues.",
    "Summarize the evidence for BHB’s effects on inflammatory bowel disease (IBD) or gut inflammation.",
    "Explain the concept of ‘BHB tolerance’ or adaptation with repeated supplementation.",
    "Describe the influence of BHB on sleep architecture or sleep quality.",
    "Summarize the potential benefits and risks of combining BHB supplementation with a ketogenic diet.",
    "Explain how BHB affects mitochondrial ROS production and antioxidant defenses.",
    "Describe the mechanism of BHB uptake into cells via monocarboxylate transporters.",
    "Summarize what is known about BHB in pediatric populations (e.g., epilepsy).",
    "Explain how BHB may modulate pain perception or nociception.",
    "Describe the role of BHB in ketosis induced by SGLT2 inhibitors.",
    "Summarize BHB’s influence on muscle protein synthesis or breakdown.",
    "Explain how chronic BHB elevation might impact long-term metabolic health.",
    "Describe the evidence for BHB’s role in multiple sclerosis or demyelinating diseases.",
    "Summarize how BHB interacts with adipokines (e.g., leptin, adiponectin).",
    "Explain the mechanisms by which BHB levels are regulated in the liver.",
    "Describe BHB’s effects on markers of oxidative DNA damage.",
    "Summarize the potential role of BHB in migraine pathophysiology.",
    "Explain how BHB interacts with immune cell subsets (e.g., T cells, macrophages).",
    "Describe the role of BHB in perioperative medicine or anesthesia contexts.",
    "Summarize how BHB might affect wound healing or tissue repair.",
    "Explain the potential interactions between BHB and commonly used medications.",
    "Describe BHB’s role in diabetic ketoacidosis versus nutritional ketosis.",
    "Summarize the relationship between BHB and cardiovascular risk markers (e.g., CRP, triglycerides).",
    "Explain how BHB may influence mitochondrial dynamics (fusion/fission).",
    "Describe the effects of BHB on mitochondrial membrane potential.",
    "Summarize the evidence for BHB as a biomarker of metabolic flexibility.",
    "Explain how BHB interacts with pyruvate dehydrogenase (PDH) activity.",
    "Describe the relationship between BHB and lactate in exercise metabolism.",
    "Summarize how BHB affects mitochondrial uncoupling proteins (UCPs).",
    "Explain how BHB may influence AMPK signaling pathways.",
    "Describe the role of BHB in the regulation of FOXO transcription factors.",
    "Summarize what is known about BHB and endothelial nitric oxide synthase (eNOS).",
    "Explain the interplay between BHB and ketone body metabolism in the heart.",
    "Describe the potential for BHB to modulate sarcopenia in older adults.",
    "Summarize the evidence for BHB’s effects on mitochondrial quality control.",
    "Explain how BHB may impact glycogen storage and utilization.",
    "Describe BHB’s effects on mitochondrial permeability transition pore (mPTP) opening.",
    "Summarize the role of BHB in reperfusion injury after ischemic events.",
    "Explain how BHB may modulate NF-κB signaling in different tissues.",
    "Describe the relationship between BHB and telomere length or cellular senescence.",
    "Summarize how BHB influences mitochondrial NADH/NAD+ ratios.",
    "Explain how BHB impacts peroxisome proliferator-activated receptors (PPARs).",
    "Describe the role of BHB in the context of ketogenic therapies for neurological disorders.",
    "Summarize what is known about BHB’s effects on mitochondrial supercomplex formation.",
    "Explain how BHB may influence ferroptosis or other non-apoptotic cell death pathways.",
    "Describe the potential for BHB as an adjunct therapy in critical care nutrition.",
    "Summarize how BHB affects mitochondrial calcium handling.",
    "Explain how BHB may modulate succinate dehydrogenase (SDH) activity.",
    "Describe the interplay between BHB and ketolytic enzymes in various tissues.",
    "Summarize the evidence for BHB in traumatic brain injury models or patients.",
    "Explain how BHB may influence microglial activation and neuroinflammation.",
    "Describe the relationship between BHB and brain-derived neurotrophic factor (BDNF).",
    "Summarize how BHB levels correlate with cognitive test outcomes in clinical trials.",
    "Explain the role of BHB in metabolic reprogramming of immune cells.",
    "Describe how BHB may modulate mitochondrial transcription and replication.",
    "Summarize the interactions between BHB and glutamate/GABA neurotransmission.",
    "Explain the role of BHB in the context of ketogenic diets used for cancer therapy.",
    "Describe BHB’s effects on autophagy in different cell types.",
    "Summarize how BHB may modulate mitophagy and mitochondrial turnover.",
    "Explain the interactions between BHB and reactive nitrogen species (RNS).",
    "Describe the potential for BHB to influence frailty or functional status in older adults.",
    "Summarize what is known about BHB’s effects on lipid peroxidation markers.",
    "Explain how BHB may modulate T cell exhaustion or memory formation.",
    "Describe the role of BHB in renal gluconeogenesis during fasting.",
    "Summarize how BHB interacts with IGF-1 signaling pathways.",
    "Explain the potential for BHB to modulate brown adipose tissue activation.",
    "Describe BHB’s effects on mitochondrial cristae structure and function.",
    "Summarize the role of BHB in the regulation of appetite-related hormones (ghrelin, peptide YY).",
    "Explain how BHB may modulate ER stress and the unfolded protein response.",
    "Describe BHB’s impact on mitochondrial antioxidant enzyme expression.",
    "Summarize how BHB might contribute to resilience against metabolic stress.",
    "Explain the potential interactions between BHB and sex hormones in metabolic regulation.",
    "Describe BHB’s role in the context of metabolic syndrome and its components.",
    "Summarize how BHB may modulate macrophage polarization (M1/M2).",
    "Explain how BHB influences the balance between oxidative and glycolytic muscle fibers.",
    "Describe BHB’s role in regulating hepatic gluconeogenesis during prolonged fasting.",
    "Summarize the interactions between BHB and the hypothalamic–pituitary–adrenal (HPA) axis.",
    "Explain how BHB may modulate endothelial progenitor cell function.",
    "Describe BHB’s effects on markers of cellular energy charge (ATP/ADP/AMP).",
    "Summarize the potential role of BHB in mood disorders such as depression and bipolar disorder.",
    "Explain how BHB may affect mitochondrial DNA integrity and repair.",
    "Describe how BHB interacts with one-carbon metabolism and methylation pathways.",
    "Summarize the evidence for BHB’s role in appetite control via hypothalamic circuits.",
    "Explain how BHB may modulate circadian rhythms and clock gene expression.",
    "Describe the role of BHB in the adaptive response to prolonged exercise.",
    "Summarize the interactions between BHB and glutathione metabolism.",
    "Explain how BHB may influence mitochondrial ATP synthase activity.",
    "Describe the potential for BHB to protect against ischemic stroke.",
    "Summarize how BHB may modulate mitochondrial fusion/fission proteins (MFN, OPA1, DRP1).",
    "Explain the role of BHB in regulating inflammatory mediators in adipose tissue.",
    "Describe the interactions between BHB and key regulators of mitochondrial biogenesis (NRF1, TFAM).",
    "Summarize what is known about BHB’s effects on proteostasis and protein turnover.",
    "Explain how BHB may modulate neuronal excitability and seizure thresholds.",
    "Describe BHB’s role in regulating hepatic ketogenesis during different nutritional states.",
    "Summarize the potential of BHB as a therapeutic agent in age-related cognitive decline.",
    "Explain how BHB may modulate mitochondrial NADPH production.",
    "Describe the interactions between BHB and AMPK–mTOR signaling in skeletal muscle.",
    "Summarize how BHB may impact mitochondrial cardiolipin content and function.",
    "Explain the role of BHB in regulating mitochondrial biogenesis in the brain.",
    "Describe the potential for BHB to modulate the gut–brain axis.",
    "Summarize how BHB may influence mitochondrial cytochrome c oxidase activity.",
    "Explain the interactions between BHB and peroxisomal fatty acid oxidation.",
    "Describe the role of BHB in the context of fasting-mimicking diets.",
    "Summarize how BHB may modulate immune checkpoint pathways in cancer.",
    "Explain the potential for BHB to modulate skeletal muscle regeneration after injury.",
    "Describe the interactions between BHB and key regulators of lipid metabolism (SREBP, LXR).",
    "Summarize how BHB may influence mitochondrial substrate preference in different tissues.",
    "Explain the role of BHB in regulating mitochondrial oxidative stress responses.",
    "Describe the potential for BHB to modulate neuroinflammation in psychiatric disorders.",
    "Characterize BHB’s influence on NLRP3 inflammasome and cytokines; note moderators and contradictions.",
    "Summarize BHB-linked HDAC effects on chromatin: key marks, gene programs, context.",
    "Describe exercise-related effects of BHB on performance, substrate use, and recovery.",
    "Map BHB effects on HCAR2/GPR109A signaling with downstream markers and contexts.",
    "Synthesize immune effects of BHB across contexts; mention key cytokines and cell responses.",
    "Summarize safety and tolerability signals for exogenous BHB in human studies.",
    "Integrate mitochondrial and redox effects of BHB with reported markers and tissues.",
    "List mechanisms via which BHB can alleviate chronic obstructive pulmonary disease",

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
        docs = retriever.get_relevant_documents(user_question)
        st.session_state["last_docs"] = docs
        answer = doc_chain.invoke({"question": user_question, "context": docs})
    st.subheader("Answer")
    st.write(answer)
    if not docs:
        st.info("No documents were retrieved for this question. The answer may be based on prior model knowledge.")
elif not run_button:
    st.info("Enter a question and click **Run**, or choose an example below.")
