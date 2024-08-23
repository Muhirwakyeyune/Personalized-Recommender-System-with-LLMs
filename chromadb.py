import pandas as pd
import re
import os
import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Read the CSV file containing the preprocessed documents
data = pd.read_csv("/Users/salomonmuhirwa/Desktop/NLP_System/preprocessed_dataset.csv")

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)  
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))  # Use stopwords from NLTK
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Preprocess the document text in the dataset
data['preprocessed_text'] = data['document'].apply(preprocess_text)

# Fit TF-IDF vectorizer on preprocessed text
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_text'])

# Function to retrieve top 3 similar documents based on user query
def retrieve_top_similar_documents(user_query):
    # Preprocess user query
    preprocessed_query = preprocess_text(user_query)
    
    # Transform user query into TF-IDF vector
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    
    # Calculate cosine similarity between user query and all documents
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top 3 documents with highest similarity scores
    top_similar_indices = similarities.argsort()[-3:][::-1]
    
    # Retrieve top 3 similar documents and their similarity scores
    top_similar_documents = []
    for idx in top_similar_indices:
        document = data.iloc[idx]['document']
        similarity_score = similarities[idx]
        top_similar_documents.append((document, similarity_score))
    
    return top_similar_documents

# Configure Generative AI
genai.configure(api_key=os.getenv("google_api_key"))

# Function to initialize chat with Gemini model
def initialize_chat():
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    return chat

# Function to get response from Gemini model
def get_gemini_response(chat, question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Document Similarity Search and Question Answering")
st.title("SightChat")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Text input and submit button for user query
input_text = st.text_input("Input: ", key="input")
submit_button = st.button("Ask the question")

# Process user input and generate response
if submit_button and input_text:
    # Retrieve top 3 similar documents based on user query
    top_similar_documents = retrieve_top_similar_documents(input_text)
    
    # Initialize chat with Generative AI model
    chat_instance = initialize_chat()
    
    # Get response from Generative AI model
    try:
        response = get_gemini_response(chat_instance, input_text)
        model_response = ""
        for chunk in response:
            try:
                text = chunk.text
                if text:
                    model_response += text + " "
                else:
                    st.session_state['chat_history'].append(("ChatSDG", "Error: Empty response"))
            except (ValueError, AttributeError):
                st.session_state['chat_history'].append(("ChatSDG", "Error: Invalid response"))
        
        # Update chat history with user query, top similar documents, and model response
        st.session_state['chat_history'].append(("You", input_text))
        st.session_state['chat_history'].append(("Similar Documents", ""))
        for document, similarity_score in top_similar_documents:
            st.session_state['chat_history'].append(("Similar Documents", f"Document: {document}\nSimilarity Score: {similarity_score}"))
        st.session_state['chat_history'].append(("Model", model_response))

        # Display chat history
        st.subheader("Chat History")
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
