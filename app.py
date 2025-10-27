# app.py
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

# MODIFIED: Import libraries for document loading
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

# ==============================================================================
# --- 1. LANGGRAPH RAG LOGIC ---
# The core graph logic remains the same.
# ==============================================================================

load_dotenv()

# --- Define the State for our Graph ---
class GraphState(TypedDict):
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]
    sources: List[str]

# --- Setup LLM (can be defined globally) ---
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", streaming=True) # MODIFIED: Updated model name for clarity
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --- Define the Graph Nodes ---
# MODIFIED: The retriever is no longer global. It will be passed in the state.
def retrieve_documents(state: GraphState):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    
    # Retrieve documents from the session-specific retriever
    # This requires us to create the retriever first and store it.
    # We will handle this in the Streamlit UI section.
    documents = st.session_state.retriever.invoke(question)
    
    doc_contents = [doc.page_content for doc in documents]
    sources = [doc.metadata.get("source", "N/A") for doc in documents]
    return {"documents": doc_contents, "question": question, "sources": sources}

def rewrite_query(state: GraphState):
    print("---REWRITING QUERY---")
    question = state["question"]
    chat_history = state["chat_history"]
    if not chat_history:
        return {"question": question}
    
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at rephrasing a follow-up question to be a standalone question, using the context of a chat history."),
            ("placeholder", "{chat_history}"),
            ("human", "Based on the chat history, rephrase the following follow-up question into a standalone question.\n"
                      "Your ONLY job is to rephrase the question. DO NOT answer it.\n"
                      "Follow-up Question: {question}"),
        ]
    )
    rewriter = rewrite_prompt | llm | StrOutputParser()
    rewritten_question = rewriter.invoke({"chat_history": chat_history, "question": question})
    print(f"Rewritten question: {rewritten_question}")
    return {"question": rewritten_question}

def generate_response_rag(state: GraphState):
    print("---GENERATING RAG RESPONSE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    context_text = "\n\n---\n\n".join(documents)
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. Answer the user's question based on the following context and the conversation history.\n\nContext:\n{context}"), ("placeholder", "{chat_history}"), ("human", "{question}")])
    chain = prompt_template | llm | StrOutputParser()
    return {"generation": chain.stream({"context": context_text, "chat_history": chat_history, "question": question})}

def generate_conversational_response(state: GraphState):
    print("---GENERATING CONVERSATIONAL RESPONSE---")
    question = state["question"]
    chat_history = state["chat_history"]
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful and friendly chatbot. Respond to the user's message conversationally."), ("placeholder", "{chat_history}"), ("human", "{question}")])
    chain = prompt_template | llm | StrOutputParser()
    return {"generation": chain.stream({"chat_history": chat_history, "question": question}), "sources": []}

def route_question(state: GraphState):
    print("---ROUTING QUESTION---")
    question = state["question"]
    prompt_template = ChatPromptTemplate.from_template("""Given the user's question below, classify it as either "rag" or "conversational".
    Do not respond with more than one word.

    - "rag": For questions that require specific information from a knowledge base.
    - "conversational": For greetings, thank yous, or other conversational filler.

    Question: {question}
    Classification:""")
    router_chain = prompt_template | llm | StrOutputParser()
    result = router_chain.invoke({"question": question})
    print(f"Route: {result}")
    if "conversational" in result.lower():
        return "conversational"
    else:
        return "rag"

# --- Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_rag_response", generate_response_rag)
workflow.add_node("generate_conversational_response", generate_conversational_response)

workflow.set_conditional_entry_point(
    route_question,
    {"rag": "rewrite_query", "conversational": "generate_conversational_response"},
)

workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_rag_response")
workflow.add_edge("generate_rag_response", END)
workflow.add_edge("generate_conversational_response", END)

app = workflow.compile()

# ==============================================================================
# --- 2. STREAMLIT UI (DYNAMIC DOCUMENT CHAT) ---
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Your Document",
    page_icon="📄",
    layout="centered"
)

# --- Helper Function for Processing Uploaded File ---
@st.cache_resource(show_spinner="Processing document...")
def process_document(file):
    """
    Loads, splits, and creates a vector store and retriever for the uploaded file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Load the document
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create an IN-MEMORY vector store
        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function)
        
        # Create retriever
        retriever = vector_store.as_retriever(k=3)
        return retriever
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# --- Sidebar ---
with st.sidebar:
    st.title("📄 Chat with Your Document")
    st.markdown("""
    Upload a PDF document and ask any questions about its content. 
    The app uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers.
    """)
    
    # MODIFIED: File uploader in the sidebar
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    st.markdown("---")
    if "retriever" in st.session_state:
        if st.button("Start New Chat", use_container_width=True, type="primary"):
            # Clear all session state to start fresh
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Main Content ---

st.title("Chat with Your Document 📄")

# MODIFIED: Main application logic flow
if uploaded_file:
    # Process the document and create the retriever if it doesn't exist
    if "retriever" not in st.session_state:
        st.session_state.retriever = process_document(uploaded_file)
        # Store the name of the uploaded file
        st.session_state.uploaded_file_name = uploaded_file.name

    st.info(f"Currently chatting with: **{st.session_state.uploaded_file_name}**")

    # Display Chat History
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            def stream_response_generator():
                inputs = {
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                }
                for event in app.stream(inputs):
                    node_name = list(event.keys())[0]
                    if node_name in ["generate_rag_response", "generate_conversational_response"]:
                        generation_stream = event[node_name]['generation']
                        for chunk in generation_stream:
                            yield chunk
            
            full_response = st.write_stream(stream_response_generator)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })

            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=full_response)
            ])
else:
    st.info("Please upload a PDF document in the sidebar to begin.")
