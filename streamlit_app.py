import logging
import json
from io import BytesIO
from PyPDF2 import PdfReader
import os
import streamlit as st
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





st.write(1)





#OPENAIKEY
os.environ['OPENAI_API_KEY'] = "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"

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
reader = ocr_pdf_with_options('./FS 1.1.pdf', './pdfservices-api-credentials.json')
pdf_stream = BytesIO(reader)
raw_text = ''
pdf_reader = PdfReader(pdf_stream)

try:
    pdf_reader = PdfReader(pdf_stream)
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
except Exception as e:
    print(f"Error processing PDF: {e}")
# print(raw_text)
#Split the extracted text to chunks
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
chunks = text_splitter.split_text(raw_text)
#Embed the text
embeddings = OpenAIEmbeddings()
VectorStore = FAISS.from_texts(chunks, embeddings)
##################Langchain Operations####################
retriever = VectorStore.as_retriever()
# chat completion llm
llm = ChatOpenAI(
    model_name='gpt-4',
    temperature=0.7
)
# conversational memory
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    callbacks=None
)

#result = qa.invoke(query)
#print(result)
knowledge_tool = Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering questions to get '
            'more information about the financial values'
        )
    )
problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(name="Calculator",
                               func=problem_chain.run,
                               description="Useful for when you need to answer numeric questions. This tool is "
                                           "only for math questions and nothing else. Only input math "
                                           "expressions, without text",
                               )
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[knowledge_tool, math_tool],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)
query = "What is the percentage increase in total fixed assets since previous year?"
result = agent(query)
st.write(result)
 
