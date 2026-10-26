from typing import Any
import httpx2 as httpx
import streamlit as st
from app.config import get_settings

settings = get_settings()
BACKEND_CHAT_URL = f"{settings.BACKEND_API_URL.rstrip('/')}/chat"

st.set_page_config(
    page_title="Microservices Docs RAG",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Microservices Documentation RAG")
st.caption("Ask questions across your microservice markdown specifications (Auth, Billing, etc.)")

# Sidebar for optional microservice filter & retrieval parameters
with st.sidebar:
    st.header("Search Filters")
    selected_service = st.selectbox(
        "Filter by Microservice",
        options=["All Services", "billing", "auth"],
        index=0,
    )
    top_k = st.slider("Passages to retrieve (top_k)", min_value=1, max_value=10, value=3)
    rag_mode = st.radio(
        "RAG mode",
        options=["vector", "graph", "hybrid"],
        index=0,
        help="vector = embeddings search in Qdrant; graph = knowledge-graph retrieval via LlamaIndex PropertyGraphIndex (Neo4j); hybrid = vector chunks with Neo4j graph facts appended to the context",
    )
    st.caption("vector finds passages by embedding similarity; graph follows relationships in the knowledge graph; hybrid combines both (vector-first, graph facts appended).")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome! Ask me anything about your microservices architecture, endpoints, or authentication.\n\n"
                "**Example queries:**\n"
                "- *How do I exchange client credentials in Auth Service?*\n"
                "- *What is the endpoint to create a customer subscription in Billing?*\n"
                "- *How are JWT signatures verified?*"
            ),
            "sources": [],
        }
    ]

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        sources: list[dict[str, Any]] = msg.get("sources", [])
        if sources:
            with st.expander("📚 Provenance Sources"):
                for s in sources:
                    st.markdown(
                        f"**`{s['service_name']}`** (chunk #{s['chunk_index']}, "
                        f"score: `{s['similarity_score']:.3f}`)\n\n"
                        f"> {s['snippet']}"
                    )

# Handle new user input
if prompt := st.chat_input("Ask a question about your microservices..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload: dict[str, Any] = {
        "message": prompt,
        "top_k": top_k,
        "retrieve_mode": rag_mode,
    }
    if selected_service != "All Services":
        payload["service_filter"] = selected_service

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation & querying LLM..."):
            try:
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(BACKEND_CHAT_URL, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    reply_text: str = data.get("reply", "No response generated.")
                    sources_list: list[dict[str, Any]] = data.get("sources", [])

                    st.markdown(reply_text)
                    if sources_list:
                        with st.expander("📚 Provenance Sources"):
                            for s in sources_list:
                                st.markdown(
                                    f"**`{s['service_name']}`** (chunk #{s['chunk_index']}, "
                                    f"score: `{s['similarity_score']:.3f}`)\n\n"
                                    f"> {s['snippet']}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply_text,
                            "sources": sources_list,
                        }
                    )
                else:
                    err_msg = f"⚠️ Backend error ({resp.status_code}): {resp.text}"
                    st.error(err_msg)
            except Exception as exc:
                st.error(f"❌ Failed to reach backend API at `{BACKEND_CHAT_URL}`: {exc}")
