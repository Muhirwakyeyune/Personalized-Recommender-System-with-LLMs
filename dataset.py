# 
import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import textract  # For extracting text from .doc and .docx files
import logging
import zipfile
import xml.etree.ElementTree as ET

# Define the path to your dataset
dataset_path = '/Users/salomonmuhirwa/Desktop/NLP_System/datasetw'

# Setup logging for the entire script
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
    return text

# Function to extract text from .doc and .docx files
def extract_text_from_doc(doc_path):
    text = ""
    try:
        text = textract.process(doc_path, output_encoding='utf-8').decode('utf-8')
    except Exception as e:
        logging.error(f"Error extracting text from {doc_path}: {str(e)}")
    return text

# Function to extract data from Excel files
def extract_data_from_excel(excel_path):
    df = pd.DataFrame()
    try:
        # Specify the engine as 'openpyxl' to handle newer Excel formats
        df = pd.read_excel(excel_path, engine='openpyxl')
    except Exception as e:
        logging.error(f"Error reading Excel file {excel_path}: {str(e)}")
    return df.to_string(index=False)

# Function to extract text from XMind files
def extract_text_from_xmind(xmind_path):
    text = ""
    try:
        with zipfile.ZipFile(xmind_path, 'r') as zip_ref:
            # Extract content.xml from the XMind file
            for file in zip_ref.namelist():
                if file.endswith('content.xml'):
                    content = zip_ref.read(file)
                    root = ET.fromstring(content)
                    for node in root.iter():
                        if node.text:
                            text += node.text.strip() + ' '
    except Exception as e:
        logging.error(f"Error extracting text from {xmind_path}: {str(e)}")
    return text

# Create a combined dataset from PDF, .doc, .docx, .xls, .xlsx, and .xmind files
def create_combined_dataset(dataset_path):
    combined_data = []
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        # Skip temporary files starting with '~$'
        if filename.startswith('~$'):
            logging.warning(f"Skipping temporary file {file_path}")
            continue
        if filename.endswith('.pdf'):
            pdf_text = extract_text_from_pdf(file_path)
            combined_data.append({'filename': filename, 'content': pdf_text})
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            excel_data = extract_data_from_excel(file_path)
            combined_data.append({'filename': filename, 'content': excel_data})
        elif filename.endswith('.doc') or filename.endswith('.docx'):
            doc_text = extract_text_from_doc(file_path)
            combined_data.append({'filename': filename, 'content': doc_text})
        elif filename.endswith('.xmind'):
            xmind_text = extract_text_from_xmind(file_path)
            combined_data.append({'filename': filename, 'content': xmind_text})
        else:
            logging.warning(f"Skipping file {file_path} as it is not a supported file type.")
    return pd.DataFrame(combined_data)

# Create the combined dataset
combined_data_df = create_combined_dataset(dataset_path)

# Save the combined dataset to a CSV file
csv_path = '/Users/salomonmuhirwa/Desktop/NLP_System/combined_dataset.csv'
combined_data_df.to_csv(csv_path, index=False)

print(f"Dataset successfully created and saved to {csv_path}")
