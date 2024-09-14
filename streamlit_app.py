import logging
import json
from io import BytesIO
from PyPDF2 import PdfReader
import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
# from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
 
from langchain.agents import Tool, initialize_agent, OpenAIFunctionsAgent, AgentExecutor
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

# from functions import *

#OPENAIKEY
api_key = "sk-svcacct-1zPs3N4CuXrMJgUKFmOz0GBKT3iYpz6q9xXazC8pwaO17jvFUW7I9lIVO86sqQ1T3BlbkFJCiN4obeFtNBA7ggV6_P4txg2trDGM6MGolx8I18SrcNqOce1AquOz-QMRfX8W2gA"
os.environ['OPENAI_API_KEY'] = api_key



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



with st.sidebar:
    st.write('123')




st.markdown(f"""<div style="border-radius: 5px;"><h2 style="text-align:left; color: white; ">Chat with PDF</h2></div>""", unsafe_allow_html=True)

fs_pdf_file = st.file_uploader('Upload Financial Statement', type = 'pdf')


if fs_pdf_file:
 reader = ocr_pdf_with_options('./Financial_Stmts_CONSO_USD.pdf', './pdfservices-api-credentials.json')
  
 pdf_stream = BytesIO(reader)
  
 raw_text_financials = ''
  
 pdf_reader = PdfReader(pdf_stream)
  
  
 try:
     pdf_reader = PdfReader(pdf_stream)
  
     for i, page in enumerate(pdf_reader.pages):
         text = page.extract_text()
         if text:
             raw_text_financials += text
  
 except Exception as e:
     st.write(f"Error processing PDF: {e}")
  
 # st.write(raw_text)
  
 #Split the extracted text to chunks
 text_splitter = CharacterTextSplitter(
     separator = "\n",
     chunk_size = 1000,
     chunk_overlap  = 200,
     length_function = len,
 )
 chunks = text_splitter.split_text(raw_text_financials)
  
 #Embed the text
 embeddings = OpenAIEmbeddings()
 VectorStore = FAISS.from_texts(chunks, embeddings)
 store_name = "Stmts"
  
 # if os.path.exists(f"{store_name}.pkl"):
 #     with open(f"{store_name}.pkl", "rb") as f:
 #         VectorStore=pickle.load(f)
 # else:
 #     embeddings = OpenAIEmbeddings()
 #     VectorStore = FAISS.from_texts(chunks, embeddings)
 #     with open(f"{store_name}.pkl", "wb") as f:
 #         pickle.dump(VectorStore, f)
  
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
  
 #Chat Prompt Template
 chat_prompt = ChatPromptTemplate(
     input_variable=["input", "messages"],
     messages=[
         MessagesPlaceholder(variable_name="chat_history"),
         HumanMessagePromptTemplate.from_template("{input}"),
         MessagesPlaceholder(variable_name="agent_scratchpad")
     ]
 )
  
 # retrieval qa chain
 qa = RetrievalQA.from_chain_type(
     llm=llm,
     chain_type="stuff",
     retriever=retriever,
     callbacks=None
 )
  
  
 #result = qa.invoke(query)
 #st.write(result)
 # Tools
 knowledge_tool = Tool(
         name='KnowledgeBase',
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
     prompt=chat_prompt,
     tools=[knowledge_tool, math_tool],
     llm=llm,
     verbose=True,
     #max_iterations=3,
     #early_stopping_method='generate',
     memory=conversational_memory
 )
  
 #Prompts Output
 df=pd.DataFrame()
 query = "What is the name of the company? Return one word answer."
 result = agent({"input":query})
 #st.write(result)
 df['Company Name'] = [result["output"]]
  
 query = "What is the latest date of the financial statement? Return one word answer"
 result = agent({"input":query})
 #st.write(result)
 df['As of Date'] = [result["output"]]
  
 query = "What are the latest current assets as of this date? Return one word answer"
 result = agent({"input":query})
 df['Current Assets'] = [result["output"]]
 #st.write(result)
  
 query = "What are the latest current liabilities as of the financial statement? Return one word answer"
 result = agent({"input":query})
 df['Current Liabilities'] = [result["output"]]
 #st.write(result)
  
 current_assets = df['Current Assets'].iloc[0]
 current_liabilities = df['Current Liabilities'].iloc[0]
 query = f"The current assets are {current_assets} and the current liabilities are {current_liabilities}. What is the current ratio? Return one number onlyr"
 result = agent({"input":query})
 df['Current Ratio'] = [result["output"]]
 #st.write(result)
  
 query = "What is the latest total Tangible Net worth as of the financial statement? Return one word answer"
 result = agent({"input":query})
 df['Net Worth'] = [result["output"]]
 #st.write(result)
  
 query = "What is the latest EBIDTA to Interest Expense + CPLTD as of the financial statement? Return one word answer"
 result = agent({"input":query})
 df['EBIDTA/Int+CPLTD'] = [result["output"]]
 #st.write(result)
  
 query = "What is the latest Debt to Net worth as of the financial statement? Return one word answer"
 result = agent({"input":query})
 df['Debt/Net Worth'] = [result["output"]]
 #st.write(result)
  
 st.write(df)


























































































# # Process the two PDF files
# cols = st.columns(2)
# with cols[0]:
#     fs_pdf_file = st.file_uploader('Upload Financial Statement', type = 'pdf')
# with cols[1]:
#     cov_pdf_file = st.file_uploader('Upload Covenants', type = 'pdf')


# if (fs_pdf_file is not None) and (cov_pdf_file is not None): 
#     pdf_files = [fs_pdf_file, cov_pdf_file]  # Update with actual file paths
#     raw_text = extract_text_from_pdfs(pdf_files)
    
    
#     # Split the extracted text into chunks
#     text_splitter = CharacterTextSplitter(
#         separator = "\n",
#         chunk_size = 1000,
#         chunk_overlap  = 200,
#         length_function = len,
#     )
#     chunks = text_splitter.split_text(raw_text)
    
#     # Embed the text
#     embeddings = OpenAIEmbeddings(api_key=api_key)
#     VectorStore = FAISS.from_texts(chunks, embeddings)
    
#     ##################Langchain Operations####################
#     retriever = VectorStore.as_retriever()
    
#     # Chat completion llm
#     llm = ChatOpenAI(
#         model_name='gpt-4',
#         temperature=0.7
#     )
    
#     # Conversational memory
#     conversational_memory = ConversationBufferMemory(
#         memory_key='chat_history',
#         chat_memory=FileChatMessageHistory('messages.json'),
#         return_messages=True
#     )
    
#     # Retrieval QA chain
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         callbacks=None
#     )
    
#     # Tools
#     knowledge_tool = Tool(
#         name='Knowledge Base',
#         func=qa.run,
#         description=(
#             'use this tool when answering questions to get '
#             'more information about the financial values'
#         )
#     )

#     # covenant_tool = Tool(
#     #     name='Covenant Base',
#     #     func=qa.run,
#     #     description=(
#     #         'use this tool when answering questions to get '
#     #         'more information about the covenants and their thresholds'
#     #     )
#     # )
    
#     problem_chain = LLMMathChain.from_llm(llm=llm)
#     math_tool = Tool.from_function(name="Calculator",
#                                    func=problem_chain.run,
#                                    description="Useful for when you need to answer numeric questions. This tool is "
#                                                "only for math questions and nothing else. Only input math "
#                                                "expressions, without text",
#                                    )
    
#     # Initialize the agent
#     agent = initialize_agent(
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         tools=[knowledge_tool, math_tool],
#         llm=llm,
#         verbose=True,
#         # max_iterations=3,
#         early_stopping_method='generate',
#         memory=conversational_memory
#     )
    
#     # Streamlit interface to ask a query
#     query_input = st.text_input("Ask a question", value="")
#     if query_input:
#         result = agent.run(query_input)
#         st.write(result)











# ##################Upload PDFs####################
# pdf_file = st.file_uploader('Upload a PDF File', type = 'pdf')

# if pdf_file:
#     # reader = ocr_pdf_with_options('./FS 1.1.pdf', './pdfservices-api-credentials.json')
#     reader = ocr_pdf_with_options(pdf_file, './pdfservices-api-credentials.json')
    
#     pdf_stream = BytesIO(reader)
    
#     raw_text = ''
#     pdf_reader = PdfReader(pdf_stream)
    
#     try:
#         pdf_reader = PdfReader(pdf_stream)
#         for i, page in enumerate(pdf_reader.pages):
#             text = page.extract_text()
#             if text:
#                 raw_text += text
#     except Exception as e:
#         st.write(f"Error processing PDF: {e}")

#     #Split the extracted text to chunks
#     text_splitter = CharacterTextSplitter(
#         separator = "\n",
#         chunk_size = 1000,
#         chunk_overlap  = 200,
#         length_function = len,
#     )
#     chunks = text_splitter.split_text(raw_text)
#     #Embed the text
#     embeddings = OpenAIEmbeddings(api_key = api_key)
#     VectorStore = FAISS.from_texts(chunks, embeddings)
    
#     ##################Langchain Operations####################
#     retriever = VectorStore.as_retriever()
#     # chat completion llm
#     llm = ChatOpenAI(
#         model_name='gpt-4',
#         temperature=0.7
#     )
#     # conversational memory
#     conversational_memory = ConversationBufferMemory(
#         memory_key='chat_history',
#         return_messages=True
#     )
#     # retrieval qa chain
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         callbacks=None
#     )
    
#     #result = qa.invoke(query)
#     #st.write(result)
#     knowledge_tool = Tool(
#             name='Knowledge Base',
#             func=qa.run,
#             description=(
#                 'use this tool when answering questions to get '
#                 'more information about the financial values'
#             )
#         )
#     problem_chain = LLMMathChain.from_llm(llm=llm)
#     math_tool = Tool.from_function(name="Calculator",
#                                    func=problem_chain.run,
#                                    description="Useful for when you need to answer numeric questions. This tool is "
#                                                "only for math questions and nothing else. Only input math "
#                                                "expressions, without text",
#                                    )
#     agent = initialize_agent(
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         tools=[knowledge_tool, math_tool],
#         llm=llm,
#         verbose=True,
#         max_iterations=3,
#         early_stopping_method='generate',
#         memory=conversational_memory
#     )
#     # query = "What is the percentage increase in total fixed assets and total liabilities since previous year?"
    
#     query_input = st.text_input("Ask a question", value="")
#     if query_input:
#         result = agent.run(query_input)
#         st.write(result)
    
    # query = "What are the current assets for the recent most year in the document?"
    # result = agent.run(query)
    # st.write(result)
    
    # query = "What are the current liabilities for the recent most year in the document?"
    # result = agent.run(query)
    # st.write(result)
     
    # query = "Now calculate the current ratio using the above current assets and liabilities."
    # result = agent.run(query)
    # st.write(result)
 
