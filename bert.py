import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_fixed
import fitz  # PyMuPDF

# Function to load and preprocess data
@st.cache_data(hash_funcs={faiss.IndexFlatIP: lambda _: None})
def load_data(csv_path):
    try:
        pdf = pd.read_csv(csv_path)
        pdf['content'] = pdf['content'].astype(str)
        return pdf
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to create embeddings
@st.cache_data(hash_funcs={faiss.IndexFlatIP: lambda _: None})
def create_embeddings(_model, texts):
    try:
        embeddings = _model.encode(texts, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

# Function to setup FAISS index
@st.cache_data(hash_funcs={faiss.IndexFlatIP: lambda _: None})
def setup_index(embeddings):
    try:
        index = faiss.IndexFlatIP(len(embeddings[0]))
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"Error setting up FAISS index: {str(e)}")
        return None

# Function to search similar documents
@st.cache_data(hash_funcs={faiss.IndexFlatIP: lambda _: None})
def search_documents(query, _model, index, pdf, k=3):
    try:
        query_embedding = _model.encode([query])
        faiss.normalize_L2(query_embedding)
        top_k = index.search(query_embedding, k)
        ids = top_k[1][0].tolist()
        similarities = top_k[0][0].tolist()
        results = pdf.iloc[ids].copy()
        results["similarities"] = similarities
        return results
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return pd.DataFrame()

# Function to get response from Gemini model with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  # Retry up to 3 times with 2 seconds interval
def get_gemini_response(chat, question):
    response = chat.send_message(question, stream=True)
    return response

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Initialize Streamlit app
st.set_page_config(page_title="Document Similarity Search and Question Answering")
st.title("SightChat")

# Load data
csv_path = '/Users/salomonmuhirwa/Desktop/NLP_System/preprocessed_dataset.csv'
pdf = load_data(csv_path)

# Initialize SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/Users/salomonmuhirwa/Desktop/NLP_System/cache folder")

# Configure Generative AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/salomonmuhirwa/Desktop/NLP_System/key.json"
genai.configure(api_key=os.getenv("google_api_key"))

# Function to initialize chat with Gemini model
def initialize_chat():
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    return chat

# Create embeddings and setup FAISS index
if pdf is not None:
    texts = pdf['content'].tolist()
    embeddings = create_embeddings(model, texts)
    if embeddings is not None:
        index = setup_index(embeddings)
    else:
        index = None
else:
    embeddings = None
    index = None

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None and index is not None:
    # Extract text from uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        # Search for similar documents in the dataset
        results = search_documents(pdf_text, model, index, pdf)
        st.write("Similar Documents:")
        st.write(results)

        # Initialize generated_answer
        generated_answer = ""

        # Generate answers and recommendations using Gemini model
        chat_instance = initialize_chat()
        try:
            response = get_gemini_response(chat_instance, f"Is this document relevant? {pdf_text}")
            for chunk in response:
                text = chunk.text
                if text:
                    generated_answer += text + " "
                else:
                    st.error("Error: Empty response from Gemini model")
        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.write("Generated Answer and Recommendations:")
        st.write(generated_answer)
else:
    if uploaded_file is None:
        st.warning("Please upload a PDF file.")
    if index is None:
        st.warning("FAISS index is not set up correctly.")

# Search query input
query = st.text_input("Enter query or question:", "")

if st.button("Search"):
    if query and index is not None:
        # Document similarity search
        results = search_documents(query, model, index, pdf)
        st.write("Similar Documents:")
        st.write(results)

        # Initialize generated_answer
        generated_answer = ""

        # Generate answers and recommendations using Gemini model
        chat_instance = initialize_chat()
        try:
            response = get_gemini_response(chat_instance, query)
            for chunk in response:
                text = chunk.text
                if text:
                    generated_answer += text + " "
                else:
                    st.error("Error: Empty response from Gemini model")
        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.write("Generated Answer and Recommendations:")
        st.write(generated_answer)
    else:
        if not query:
            st.warning("Please enter a query.")
        if index is None:
            st.warning("FAISS index is not set up correctly.")
