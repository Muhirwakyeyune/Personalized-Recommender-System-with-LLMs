import streamlit as st
import pandas as pd
import os
import google.generativeai as genai  # Assuming this is for other uses of Gemini
from tenacity import retry, stop_after_attempt, wait_fixed
import fitz  # PyMuPDF
import docx
from xmindparser import xmind_to_dict
import pytesseract
import cv2
import numpy as np

# Function to load and preprocess data
def load_data(csv_path):
    try:
        pdf = pd.read_csv(csv_path)
        pdf['content'] = pdf['content'].astype(str)
        return pdf
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to get response from Gemini model with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  # Retry up to 3 times with 2 seconds interval
def get_gemini_response(chat, question):
    response = chat.send_message(question, stream=True)
    return response

# Function to extract text from PDF (with OCR for image-based PDFs)
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()  # Get text from text-based PDFs
            if not text.strip():  # If text extraction fails, try OCR
                pix = page.get_pixmap()
                img = np.array(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n > 1:  # Convert to grayscale if it's RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Function to extract text from Excel (XLSX)
def extract_text_from_excel(excel_file):
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
        text = df.to_string()
        return text
    except Exception as e:
        st.error(f"Error extracting text from Excel: {str(e)}")
        return ""

# Function to extract text from XMIND
def extract_text_from_xmind(xmind_file):
    try:
        content_dict = xmind_to_dict(xmind_file)
        text = "\n".join([str(item) for item in content_dict])
        return text
    except Exception as e:
        st.error(f"Error extracting text from XMIND: {str(e)}")
        return ""

# Function to calculate text similarity (example using simple text overlap)
def calculate_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    overlap = words1.intersection(words2)
    similarity = len(overlap) / float(len(words1.union(words2)))  # Jaccard similarity
    return similarity

# Initialize Streamlit app
st.set_page_config(page_title="Sight studio")
st.title("RHV Chat")

# Load data
csv_path = '/Users/salomonmuhirwa/Desktop/NLP_System/preprocessed_dataset.csv'
pdf = load_data(csv_path)

# Configure Generative AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/salomonmuhirwa/Desktop/NLP_System/key.json"
genai.configure(api_key=os.getenv("google_api_key"))

# Function to initialize chat with Gemini model
def initialize_chat():
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    return chat

# Function to find similar documents in dataset
def find_similar_documents(uploaded_text, pdf_dataset):
    similarities = []
    for index, row in pdf_dataset.iterrows():
        similarity_score = calculate_similarity(row['content'], uploaded_text)
        similarities.append(similarity_score)
    return similarities

# Upload file
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "xlsx", "xmind"])

if uploaded_file is not None:
    # Extract text based on file type
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'pdf':
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == 'docx':
        extracted_text = extract_text_from_docx(uploaded_file)
    elif file_type == 'xlsx':
        extracted_text = extract_text_from_excel(uploaded_file)
    elif file_type == 'xmind':
        extracted_text = extract_text_from_xmind(uploaded_file)
    else:
        st.error("Unsupported file type.")
        extracted_text = None

    if extracted_text and pdf is not None:
        # Find similarities with dataset
        similarities = find_similar_documents(extracted_text, pdf)
        max_similarity = max(similarities) if similarities else 0.0
        
        if max_similarity > 0.5:
            st.success(f"We recommend considering this document. Maximum similarity score: {max_similarity:.2f}")
        else:
            st.warning("We did not find similar documents above the threshold.")

        # Display similar documents
        st.write("Similar Documents in Dataset:")
        similar_indices = [i for i, sim in enumerate(similarities) if sim > 0.5]
        if similar_indices:
            st.write(pdf.iloc[similar_indices])
        else:
            st.warning("No similar documents found in the dataset.")

        # Initialize generated_answer
        generated_answer = ""

        # Generate answers and recommendations using Gemini model
        if st.button("Click to See Reasons"):
            chat_instance = initialize_chat()
            try:
                response = get_gemini_response(chat_instance, "Is this document relevant? ")
                for chunk in response:
                    text = chunk.text
                    if text:
                        generated_answer += text + " "
                    else:
                        st.error("Error: Empty response from Gemini model")
            except Exception as e:
                st.error(f"Error: {str(e)}")

            st.write("See Reason:")
            st.write(generated_answer)
    else:
        st.warning("Error processing uploaded file or dataset.")
else:
    st.warning("Please upload a file.")

# Search query input
query = st.text_input("Enter query or question:", "")

if st.button("Search"):
    if query and pdf is not None:
        # Calculate similarity with each document based on query
        similarities = []
        for index, row in pdf.iterrows():
            similarity_score = calculate_similarity(row['content'], query)
            similarities.append(similarity_score)
        
        # Find top similar documents based on scores
        pdf['similarity_score'] = similarities
        results = pdf.sort_values(by='similarity_score', ascending=False).head(3)

        st.write("Search Results:")
        st.write(results)

        # Initialize generated_answer
        generated_answer = ""

        # Generate answers and recommendations using Gemini model
        if query:
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
        if pdf is None:
            st.warning("Error: Dataset not loaded.")
