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

# NEW IMPORTS
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==============================================================================
# --- 1. SETUP AND PDF PROCESSING ---
# ==============================================================================

load_dotenv()

# MODIFIED: Define the path for the persistent Chroma database
CHROMA_PATH = "chroma_multidoc"

# Setup the embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# MODIFIED: Initialize a persistent Chroma client
# We initialize it once and add documents as they are uploaded.
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# NEW: PDF Processing Function
@st.cache_resource(show_spinner="Processing PDF...")
def process_and_store_pdf(uploaded_file):
    """Processes an uploaded PDF file, splits it into chunks, and stores it in ChromaDB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load the document
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # IMPORTANT: Add filename as metadata to each chunk
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        # Add chunks to the Chroma database
        db.add_documents(chunks)
        print(f"Successfully processed and stored {len(chunks)} chunks from {uploaded_file.name}")
        return True

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return False
    finally:
        os.remove(tmp_file_path) # Clean up the temporary file

# ==============================================================================
# --- 2. LANGGRAPH RAG LOGIC ---
# ==============================================================================

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", streaming=True)

# MODIFIED: Define the State for our Graph
class GraphState(TypedDict):
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]
    sources: List[str]
    selected_file: str # NEW: To know which document to query

# --- Define the Graph Nodes ---

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

# MODIFIED: retrieve_documents Node
def retrieve_documents(state: GraphState):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    selected_file = state["selected_file"] # Get the selected filename from the state

    # Create a retriever that filters by the selected filename
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 5,
            'filter': {'source': selected_file} # This is the crucial filter!
        }
    )

    documents = retriever.invoke(question)
    doc_contents = [doc.page_content for doc in documents]
    sources = [doc.metadata.get("source", "N/A") for doc in documents]
    
    print(f"Retrieved {len(documents)} documents for source '{selected_file}'")
    return {"documents": doc_contents, "question": question, "sources": sources}

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

# --- Build the Graph (No changes here) ---
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
# --- 3. STREAMLIT UI (HEAVILY MODIFIED) ---
# ==============================================================================

st.set_page_config(page_title="Chat with Your Docs", page_icon="📄", layout="centered")

# --- Session State Management ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {} # e.g., {"doc1.pdf": {"messages": [], "history": []}}

# --- Sidebar UI ---
with st.sidebar:
    st.title("📄 Chat with Your Docs")
    st.markdown("""
    Upload one or more PDFs, select a document from the dropdown, and start asking questions!
    """)
    
    # NEW: File Uploader
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    # NEW: Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                if process_and_store_pdf(file):
                    st.session_state.processed_files.append(file.name)
                    # Initialize chat history for the new file
                    st.session_state.chat_histories[file.name] = {"messages": [], "langchain_history": []}

    # NEW: Document Selector
    if st.session_state.processed_files:
        st.session_state.selected_file = st.selectbox(
            "Select a document to chat with:",
            options=st.session_state.processed_files
        )
        
        # Clear chat for the selected document
        if st.button("Clear Chat History", key=f"clear_{st.session_state.selected_file}", use_container_width=True):
            st.session_state.chat_histories[st.session_state.selected_file] = {"messages": [], "langchain_history": []}
            st.rerun()

# --- Main Content ---
st.title("Chat with Your Documents")

if not st.session_state.selected_file:
    st.info("Please upload and select a document in the sidebar to begin.")
else:
    # Get the chat history for the currently selected file
    current_messages = st.session_state.chat_histories[st.session_state.selected_file]["messages"]
    current_lc_history = st.session_state.chat_histories[st.session_state.selected_file]["langchain_history"]

    # Display Chat History for the selected document
    for message in current_messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input(f"Ask a question about {st.session_state.selected_file}..."):
        current_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            def stream_response_generator():
                inputs = {
                    "question": prompt,
                    "chat_history": current_lc_history,
                    "selected_file": st.session_state.selected_file # Pass the selected file to the graph
                }
                for event in app.stream(inputs):
                    node_name = list(event.keys())[0]
                    if node_name in ["generate_rag_response", "generate_conversational_response"]:
                        generation_stream = event[node_name]['generation']
                        for chunk in generation_stream:
                            yield chunk
            
            full_response = st.write_stream(stream_response_generator)
            
            # Save the full response to the session state for the selected file
            current_messages.append({
                "role": "assistant", 
                "content": full_response
            })

            # Update the LangChain history format as well
            current_lc_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=full_response)
            ])
