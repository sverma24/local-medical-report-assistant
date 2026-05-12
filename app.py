from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st
from langchain_ollama import ChatOllama

from src.agents.graph import build_analysis_graph
from src.config import AppConfig
from src.ingestion.parser import parse_uploaded_file
from src.retrieval.vector_store import LocalVectorStore


st.set_page_config(
    page_title="Local Medical Report Assistant",
    page_icon=":material/health_and_safety:",
    layout="wide",
)

st.title("Local Medical Report Assistant")
st.caption(
    "Private, local-first analysis for medical lab reports using Ollama + Gemma + Chroma + LangGraph."
)
st.info(
    "Informational use only. This app does not diagnose disease or replace advice from a licensed clinician."
)


@st.cache_resource(show_spinner=False)
def get_config() -> AppConfig:
    return AppConfig()


@st.cache_resource(show_spinner=True)
def get_vector_store(config: AppConfig) -> LocalVectorStore:
    store = LocalVectorStore(config)
    store.bootstrap_knowledge_base()
    return store


def _build_llm(config: AppConfig) -> ChatOllama:
    return ChatOllama(
        model=config.llm_model,
        base_url=config.ollama_base_url,
        temperature=0.2,
    )


def _render_list(title: str, values: list[str]) -> None:
    st.subheader(title)
    if not values:
        st.write("No strong signals found.")
        return
    for value in values:
        st.write(f"- {value}")


config = get_config()
vector_store = get_vector_store(config)
llm = _build_llm(config)
graph = build_analysis_graph(llm=llm, vector_store=vector_store)

with st.sidebar:
    st.header("Runtime")
    st.write(f"LLM: `{config.llm_model}`")
    st.write(f"Embeddings: `{config.embedding_model}`")
    st.write(f"Ollama endpoint: `{config.ollama_base_url}`")
    st.write(f"Chroma dir: `{config.chroma_dir}`")

uploaded_file = st.file_uploader(
    "Upload medical report (PDF, TXT, CSV, PNG/JPG)",
    type=["pdf", "txt", "md", "csv", "png", "jpg", "jpeg", "webp", "tiff"],
)

if uploaded_file is None:
    st.stop()

try:
    report_text = parse_uploaded_file(uploaded_file)
except Exception as exc:
    st.error(f"Could not parse file: {exc}")
    st.stop()

if not report_text.strip():
    st.error("Parsed report is empty.")
    st.stop()

with st.expander("Preview extracted raw text", expanded=False):
    st.text(report_text[:5000])

run = st.button("Run Local Analysis", type="primary")
if not run:
    st.stop()

report_id = str(uuid.uuid4())
with st.spinner("Indexing report in local ChromaDB..."):
    vector_store.index_report(report_id=report_id, report_text=report_text)

with st.spinner("Running LangGraph workflow with local Gemma model..."):
    result = graph.invoke({"report_id": report_id, "report_text": report_text})

measurements = result.get("measurements", [])
output = result.get("output")
tool_trace = result.get("tool_trace", [])

if tool_trace:
    with st.expander("Gemma local function calls", expanded=False):
        for item in tool_trace:
            st.json(item)

st.subheader("Extracted Lab Values")
if measurements:
    frame = pd.DataFrame([m.model_dump() for m in measurements])
    st.dataframe(frame, use_container_width=True, hide_index=True)
else:
    st.warning(
        "No structured lab values were detected with regex extraction. Try a clearer report format."
    )

if output:
    st.subheader("Plain-Language Summary")
    st.write(output.summary or "No summary generated.")

    col1, col2 = st.columns(2)
    with col1:
        _render_list("Areas of Concern", output.areas_of_concern)
        _render_list("Nutritional Signals", output.nutritional_signals)
    with col2:
        _render_list("Doctor Follow-Up", output.doctor_followup)
        _render_list("Lifestyle Recommendations", output.lifestyle_recommendations)

    _render_list("Questions to Ask Your Doctor", output.questions_for_doctor)
    st.warning(output.disclaimer)

citations = result.get("citations", [])
if citations:
    st.caption("Retrieved context sources")
    for source in sorted(set(citations)):
        st.write(f"- {source}")
