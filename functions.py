import logging
import json
from io import BytesIO
from PyPDF2 import PdfReader
import os
import streamlit as st
import pickle
# from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMMathChain
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.ocr_pdf_job import OCRPDFJob
from adobe.pdfservices.operation.pdfjobs.params.ocr_pdf.ocr_params import OCRParams
from adobe.pdfservices.operation.pdfjobs.params.ocr_pdf.ocr_supported_locale import OCRSupportedLocale
from adobe.pdfservices.operation.pdfjobs.params.ocr_pdf.ocr_supported_type import OCRSupportedType
from adobe.pdfservices.operation.pdfjobs.result.ocr_pdf_result import OCRPDFResult


# Initialize the logger
logging.basicConfig(level=logging.INFO)
def ocr_pdf_with_options(input_pdf_path: str, credentials_path: str):
    try:
        # Read the input PDF file
        with open(input_pdf_path, 'rb') as file:
            input_stream = file.read()
 
        # Load credentials
        with open(credentials_path, 'r') as f:
            credentials_data = json.load(f)
 
        client_id = credentials_data.get('client_credentials', {}).get('client_id')
        client_secret = credentials_data.get('client_credentials', {}).get('client_secret')
 
        # Set up credentials for the PDF service
        credentials = ServicePrincipalCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
 
        # Create a PDF Services instance
        pdf_services = PDFServices(credentials=credentials)
 
        # Upload the input PDF file
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
 
        # Set OCR parameters
        ocr_pdf_params = OCRParams(
            ocr_locale=OCRSupportedLocale.EN_US,
            ocr_type=OCRSupportedType.SEARCHABLE_IMAGE
        )
 
        # Create an OCR job
        ocr_pdf_job = OCRPDFJob(input_asset=input_asset, ocr_pdf_params=ocr_pdf_params)
 
        # Submit the job and get the result
        location = pdf_services.submit(ocr_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, OCRPDFResult)
 
        # Get the resulting asset
        result_asset: CloudAsset = pdf_services_response.get_result().get_asset()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
 
        # Return the output PDF stream
        return stream_asset.get_input_stream()
 
    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        logging.exception(f'Exception encountered while executing operation: {e}')
        return None




# Function to process both PDFs
def extract_text_from_pdfs(pdf_files):
    combined_raw_text = ''
    
    for pdf_file in pdf_files:
        reader = ocr_pdf_with_options(pdf_file, './pdfservices-api-credentials.json')  # Assuming you need OCR
        pdf_stream = BytesIO(reader)
        
        try:
            pdf_reader = PdfReader(pdf_stream)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    combined_raw_text += text
        except Exception as e:
            st.write(f"Error processing PDF {pdf_file}: {e}")
    
    return combined_raw_text

















