import streamlit as st
from langchain_ollama import OllamaLLM
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from openai import APIStatusError

# Set up logging
logging.basicConfig(level=logging.ERROR)

def log_error(e):
    logging.error(f"Error: {e}")
    st.error(f"Error: {e}")

# Initialize global variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_content" not in st.session_state:
    st.session_state.uploaded_files_content = ""

def initialize_chromadb():
    try:
        os.makedirs("./chromadb", exist_ok=True)
        client = chromadb.PersistentClient(path="./chromadb")
        
        api_key = "YOUR_API_KEY"
        
        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)
        collection = client.get_or_create_collection(name="kazakhstan_constitution", embedding_function=ef)
        return collection
    except Exception as e:
        log_error(f"ChromaDB Initialization Error: {e}")
        return None

def save_to_chromadb(collection, user_input, response):
    try:
        if collection:
            collection.add(
                documents=[user_input, response],
                metadatas=[{"role": "user"}, {"role": "bot"}],
                ids=[f"user-{len(st.session_state.chat_history)}", f"bot-{len(st.session_state.chat_history)}"]
            )
    except Exception as e:
        log_error(e)

def retrieve_from_chromadb(collection, query):
    try:
        if collection:
            res = collection.query(query_texts=[query], n_results=5)
            return "\n".join(res.get("documents", [[]])[0]) if res else "No relevant results found."
        return "No relevant results found."
    except Exception as e:
        log_error(e)
        return "Error retrieving context from ChromaDB."

def process_uploaded_files(files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for file in files:
        try:
            content = file.read().decode("utf-8")
            all_chunks.extend(text_splitter.split_text(content))
        except Exception as e:
            log_error(e)
    return "\n".join(all_chunks[:5000])

def generate_response(model_name, prompt):
    try:
        llm = OllamaLLM(model=model_name, api_base="http://localhost:11434")
        return llm.invoke(prompt)
    except APIStatusError as e:
        response = getattr(e, "response", "No response available")
        body = getattr(e, "body", "No body available")
        error_message = f"API Error - Response: {response}, Body: {body}"
        log_error(error_message)
        return error_message
    except Exception as e:
        log_error(e)
        return f"Error generating response: {e}"

def main():
    st.title("AI Assistant for Kazakhstan Constitution")
    collection = initialize_chromadb()
    if not collection:
        st.warning("ChromaDB could not be initialized. Contextual memory is disabled.")

    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Choose a model:", ("llama3.2:1b", "llama3.1:8b"))

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files_content = ""
        if collection:
            try:
                all_ids = collection.get().get("ids", [])
                if all_ids:
                    collection.delete(ids=all_ids)
            except Exception as e:
                log_error(e)

    uploaded_files = st.file_uploader("Upload Constitution Files:", type="txt", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files_content = process_uploaded_files(uploaded_files)
        st.success(f"Successfully uploaded {len(uploaded_files)} file(s).")
    elif not st.session_state.uploaded_files_content and os.path.exists("kazakhstan_constitution.txt"):
        with open("kazakhstan_constitution.txt", "r", encoding="utf-8") as f:
            st.session_state.uploaded_files_content = f.read()
        st.success("Loaded constitution from local file.")

    user_input = st.text_input("Ask about the Constitution:", key="user_input")
    if user_input:
        chromadb_context = retrieve_from_chromadb(collection, user_input) if collection else ""
        file_context = st.session_state.uploaded_files_content[:2000] if st.session_state.uploaded_files_content else ""
        
        combined_context = f"User Query: {user_input}\n"
        if chromadb_context:
            combined_context += f"\n[Relevant ChromaDB Context]\n{chromadb_context}\n"
        if file_context:
            combined_context += f"\n[Relevant File Context]\n{file_context}\n"

        response = generate_response(model_choice, combined_context)
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Bot: {response}")
        if collection:
            save_to_chromadb(collection, user_input, response)

    st.write("### Chat History")
    if st.session_state.chat_history:
        st.text_area("Conversation:", value="\n".join(st.session_state.chat_history), height=300, key="chat_history_display")

if __name__ == "__main__":
    main()
