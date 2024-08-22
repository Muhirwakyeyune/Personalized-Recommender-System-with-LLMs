Overview
SightChat is a Streamlit-based application designed to assist users in finding similar documents and answering questions based on a query. It combines traditional Natural Language Processing (NLP) techniques with Generative AI to deliver accurate and relevant results. The app preprocesses documents using TF-IDF vectorization to retrieve similar texts and utilizes Google's Generative AI (Gemini model) to provide insightful answers to user queries.

Features
Document Similarity Search: Retrieve the top 3 documents most similar to the user query based on cosine similarity.
Generative AI Integration: Engage in a conversational interface powered by Google Generative AI (Gemini model) for dynamic question answering.
Text Preprocessing: Robust text cleaning, tokenization, stopword removal, and lemmatization to enhance search accuracy.
Interactive Chat Interface: View the chat history and responses in a user-friendly format.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/SightChat.git
cd SightChat
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables:

Create a .env file and add your Google API key:
env
Copy code
google_api_key=your_google_api_key_here
Run the Streamlit app:

bash
Copy code
streamlit run app.py
File Structure
app.py: Main Streamlit application file.
preprocessed_dataset.csv: Dataset containing preprocessed documents for similarity search.
models--sentence-transformers--all-MiniLM-L6-v2/: Model directory for embedding generation.
preprocessing.py: Script for preprocessing the dataset.
bert.py, chromadb.py, firstsearch.py, search.py: Additional modules used in the project.
evaluation.py, sx.py, ex.py: Evaluation and experimental scripts.
key.json, credentials.json, extreme-zephyr-426913-p1-dab67148f490.json: Credential files for accessing external APIs.
combined_dataset.csv, labelled_newscatcher_dataset.csv: Additional datasets for processing.
Usage
Enter a query in the input box and click "Ask the question".
View the top 3 similar documents retrieved based on the query.
Engage with the AI-powered chat to receive an informative response from the Gemini model.
Dependencies
Python 3.x
Streamlit
Pandas
Scikit-learn
NLTK
Google Generative AI
Future Improvements
Extend the app to support more complex queries.
Integrate additional datasets for more comprehensive search results.
Enhance the chat interface for a better user experience.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Google Generative AI for powering the conversational model.
NLTK and Scikit-learn for NLP support.
Streamlit for providing a robust interface for the application.
