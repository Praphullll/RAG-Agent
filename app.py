import streamlit as st
import uuid
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Agent")
st.caption("Upload a PDF and chat with it. Powered by Ollama + LangGraph.")

# ─────────────────────────────────────────────
# Sidebar — settings & PDF upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ollama_model = st.selectbox(
        "Ollama Model",
        ["llama3:latest", "llama3", "phi3:mini", "mistral", "tinyllama"],
        index=0,
        help="Make sure the model is pulled via: ollama pull <model>"
    )

    st.divider()
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph = None
        st.session_state.ready = False
        st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown(
        "- 🔀 **Router** classifies each message\n"
        "- 📚 **Doc questions** → retrieves PDF chunks\n"
        "- 💬 **Chat questions** → uses conversation history"
    )

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ready" not in st.session_state:
    st.session_state.ready = False

if "graph" not in st.session_state:
    st.session_state.graph = None

if "last_route" not in st.session_state:
    st.session_state.last_route = None

# ─────────────────────────────────────────────
# Build RAG graph (cached so it only runs once per PDF)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_graph(pdf_bytes: bytes, model_name: str):
    """Load PDF, embed it, build the LangGraph agent. Cached per PDF + model."""

    # Save uploaded bytes to a temp file (works on Windows + Linux)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(pdf_bytes)
    tmp_file.close()
    tmp_path = tmp_file.name

    # --- Load & chunk ---
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # --- Embed & index ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # --- LLM ---
    model = ChatOllama(model=model_name)

    # --- State ---
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        retrieved_docs: list
        question_type: str

    # --- Nodes ---
    def router_node(state: State) -> State:
        user_message = state["messages"][-1].content
        router_prompt = [
            SystemMessage(content=(
                "You are a routing assistant. Your ONLY job is to classify a user message.\n"
                "Reply with exactly one word — either 'doc_question' or 'conversation'.\n\n"
                "Reply 'doc_question' if the message asks about a document, PDF, article, "
                "or any technical topic requiring reference material.\n\n"
                "Reply 'conversation' if the message is casual chat, about the user themselves "
                "(name, preferences, etc.), a greeting, or answerable from conversation history.\n\n"
                "Do NOT explain. Output only: doc_question or conversation"
            )),
            HumanMessage(content=user_message)
        ]
        result = model.invoke(router_prompt)
        classification = result.content.strip().lower()
        question_type = "conversation" if "conversation" in classification else "doc_question"
        return {"question_type": question_type}

    def route_decision(state: State) -> Literal["retrieve", "chat"]:
        return "retrieve" if state["question_type"] == "doc_question" else "chat"

    def retrieve_node(state: State) -> State:
        query = state["messages"][-1].content
        docs = retriever.invoke(query)
        return {"retrieved_docs": docs}

    doc_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions using document context.\n"
         "Use the document context below to answer. If the context doesn't contain the answer, say so.\n\n"
         "Document context:\n{context}"),
        ("human", "{question}")
    ])

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly conversational assistant.\n"
         "Answer using the conversation history below.\n\n"
         "Conversation history:\n{history}"),
        ("human", "{question}")
    ])

    def rag_answer_node(state: State) -> State:
        context = "\n\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
        question = state["messages"][-1].content
        chain = doc_prompt | model
        response = chain.invoke({"context": context, "question": question})
        return {"messages": [response]}

    def chat_answer_node(state: State) -> State:
        prior = state["messages"][:-1]
        history = "\n".join([f"{m.type.upper()}: {m.content}" for m in prior]) if prior else "(no prior conversation)"
        question = state["messages"][-1].content
        chain = chat_prompt | model
        response = chain.invoke({"history": history, "question": question})
        return {"messages": [response]}

    # --- Graph ---
    builder = StateGraph(State)
    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rag_answer", rag_answer_node)
    builder.add_node("chat_answer", chat_answer_node)

    builder.set_entry_point("router")
    builder.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "chat": "chat_answer"})
    builder.add_edge("retrieve", "rag_answer")
    builder.add_edge("rag_answer", END)
    builder.add_edge("chat_answer", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph, len(chunks)

# ─────────────────────────────────────────────
# Handle PDF upload → build graph
# ─────────────────────────────────────────────
if uploaded_file is not None:
    if not st.session_state.ready:
        with st.spinner("📚 Reading & indexing your PDF... (this takes ~20s the first time)"):
            try:
                graph, num_chunks = build_graph(uploaded_file.read(), ollama_model)
                st.session_state.graph = graph
                st.session_state.ready = True
                st.session_state.chat_history = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.sidebar.success(f"✅ Indexed {num_chunks} chunks from your PDF!")
            except Exception as e:
                st.error(f"❌ Failed to build index: {e}\n\nMake sure Ollama is running: `ollama serve`")
else:
    st.info("👈 Upload a PDF in the sidebar to get started.")

# ─────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────
if st.session_state.ready:

    # Render existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("route"):
                route_label = "📚 Retrieved from PDF" if msg["route"] == "doc_question" else "💬 From conversation"
                st.caption(route_label)

    # Chat input
    if prompt := st.chat_input("Ask something about the document, or just chat..."):

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Run the graph
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = st.session_state.graph.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config
                    )
                    answer = result["messages"][-1].content
                    route = result.get("question_type", "doc_question")

                    st.markdown(answer)
                    route_label = "📚 Retrieved from PDF" if route == "doc_question" else "💬 From conversation"
                    st.caption(route_label)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "route": route
                    })

                except Exception as e:
                    err_msg = str(e)
                    if "connection" in err_msg.lower() or "refused" in err_msg.lower():
                        st.error("❌ Can't reach Ollama. Run `ollama serve` in your terminal first.")
                    else:
                        st.error(f"❌ Error: {err_msg}")
