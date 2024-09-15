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

from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults

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

from functions import *

#OPENAIKEY
api_key = "sk-svcacct-1zPs3N4CuXrMJgUKFmOz0GBKT3iYpz6q9xXazC8pwaO17jvFUW7I9lIVO86sqQ1T3BlbkFJCiN4obeFtNBA7ggV6_P4txg2trDGM6MGolx8I18SrcNqOce1AquOz-QMRfX8W2gA"
os.environ['OPENAI_API_KEY'] = api_key

with st.sidebar:
 cols = st.columns([0.3, 3])
 
with cols[1]:
 st.markdown(f"""<div><h2 style="text-align:left; color: white; margin-top:20px; ">LLM Chat App</h2></div>""", unsafe_allow_html=True)
 st.markdown(f"""<div><h3 style="text-align:left; color: white; ">This is an LLM-Powered App</h3></div>""", unsafe_allow_html=True)

 
 st.markdown(f"""<div><h4 style="text-align:left; color: white; margin-top: -10px; ">Financial Covenants</h4></div>""", unsafe_allow_html=True)
 st.markdown(f"""<div style="border-radius: 5px;">
  <h4 style="text-align:left; color: white;">
     <ul style="list-style-type: disc; margin-left: 1px; margin-top: -20px;">
         <li style="margin-bottom: 5px; font-size: 14px;">OCR is applied to convert image-based content from PDFs into text.</li>
         <li style="margin-bottom: 5px; font-size: 14px;">OpenAI is used to calculate financial ratios from customer financial statements.</li>
         <li style="margin-bottom: 5px; font-size: 14px;">Financial covenants are reviewed to ensure threshold compliance.</li>
         <li style="margin-bottom: 5px; font-size: 14px;">A final flag confirms if there is a material breach of the covenants.</li>
     </ul>
 </h4>
 </div>
 """, unsafe_allow_html=True)


 st.markdown(f"""<div><h4 style="text-align:left; color: white; margin-top: -15px; ">Bankruptcy Flag</h4></div>""", unsafe_allow_html=True)
 st.markdown(f"""<div style="border-radius: 5px;">
  <h4 style="text-align:left; color: white;">
     <ul style="list-style-type: disc; margin-left: 1px; margin-top: -20px;">
         <li style="margin-bottom: 5px; font-size: 14px;">Tavily is a search engine optimized for LLMs.</li>
         <li style="margin-bottom: 5px; font-size: 14px;">Aggregates and filters data from up to 20 sites per API call.</li>
         <li style="margin-bottom: 5px; font-size: 14px;">Reduces hallucinations through contextual retrieval​.</li>
     </ul>
 </h4>
 </div>
 """, unsafe_allow_html=True)

st.markdown(f"""<div><h2 style="text-align:left; color: white; ">Chat with PDF</h2></div>""", unsafe_allow_html=True)

st.markdown("""
    <style>
    button[data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
    }
    .st-cr {
        color: black;
    }
    .st-d8 {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

tabs = st.tabs(['Financial Covenants', 'Bankruptcy Flag'])

with tabs[0]:

 cols = st.columns(2)
 with cols[0]:
  fs_pdf_file = st.file_uploader('Upload Financial Statement', type = 'pdf')
 with cols[1]:
  covenants_pdf_file = st.file_uploader('Upload Financial Covenants', type = 'pdf')
 
 
 if (fs_pdf_file is not None) and (covenants_pdf_file is not None):
  reader = ocr_pdf_with_options(fs_pdf_file, './pdfservices-api-credentials.json')
   
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
      chunk_size = 700,
      chunk_overlap  = 150,
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
      model_name='gpt-4o',
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
   
  query = "What is the latest date of the financial statement? Return only the date in the format DD/MM/YYYY"
  result = agent({"input":query})
  #st.write(result)
  df['As of Date'] = [result["output"]]
  as_of_date = [result["output"]][0]

  query = "Extract the scale or multiplier in which the values are written within the financial statement? Return one word answer"
  result = agent({"input":query})
  st.write(result["output"])
  currency_scale = result["output"]
   
  query = f"What are the total current assets in {currency_scale} as of {as_of_date}? Return one word answer"
  result = agent({"input":query})
  df['Current Assets'] = [result["output"]]
  #st.write(result)
   
  query = f"What are the total current liabilities in {currency_scale} as of {as_of_date}? Return one word answer"
  result = agent({"input":query})
  df['Current Liabilities'] = [result["output"]]
  # st.write("What are the total current liabilities in {currency_scale} as of {as_of_date}? Return one word answer")
  # st.write(result)
   
  current_assets = df['Current Assets'].iloc[0]
  current_liabilities = df['Current Liabilities'].iloc[0]
  query = f"The current assets are {current_assets} and the current liabilities are {current_liabilities}. What is the current ratio? Return one number only"
  result = agent({"input":query})
  df['Current Ratio'] = [result["output"]]
  #st.write(result)
   
  query = f"What is the Tangible Net worth as of {as_of_date}? Provide the result in {currency_scale} and make sure to return one word answer only"
  result = agent({"input":query})
  df['Tangible Net Worth'] = [result["output"]]
  # st.write(result)
   
  query = f"What is the latest EBIDTA to Interest Expense + CPLTD as of {as_of_date}? Return one word answer"
  result = agent({"input":query})
  df['EBITDA to Debt Service Coverage Ratio'] = [result["output"]]
  # st.write(result)
   
  query = f"What is the latest Debt/Net worth ratio as of {as_of_date}? Return one word answer"
  result = agent({"input":query})
  df['Debt/Net Worth'] = [result["output"]]
  st.write(result)
   
  st.table(df)













  ##########################COVENANTS#################
  cov_reader = ocr_pdf_with_options(covenants_pdf_file, './pdfservices-api-credentials.json')
  cov_pdf_stream = BytesIO(cov_reader)
  raw_text_cov = ''
  try:
      pdf_reader_cov = PdfReader(cov_pdf_stream)
      for i, page in enumerate(pdf_reader_cov.pages):
          text = page.extract_text()
          if text:
              raw_text_cov += text
   
  except Exception as e:
      print(f"Error processing PDF: {e}")
   
  #Split the extracted text to chunks
  cov_chunks = text_splitter.split_text(raw_text_cov)
   
  #Embed the text
  CovVectorStore = FAISS.from_texts(cov_chunks, embeddings)
  cov_retriever = CovVectorStore.as_retriever()
   
  #Financial retriever
  financial_description = df.iloc[0].to_string()
  #embeddings = OpenAIEmbeddings()
  #financial_embedding = embeddings.embed_documents([financial_description])
   
  # Create metadata for the row
  metadata = [df.iloc[0].to_dict()]
   
  # Create a FAISS vector store with the embedding and metadata
  financial_vector_store = FAISS.from_texts([financial_description], embeddings)
   
  fin_retriever = financial_vector_store.as_retriever()
   
  cov_qa = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=cov_retriever,
      callbacks=None
  )
   
  financial_qa_chain = RetrievalQA.from_chain_type(
  llm=llm,
      chain_type="stuff",
      retriever=fin_retriever,
      callbacks=None
  )
   
  # Cov Tools
  cov_knowledge_tool = Tool(
          name='CovenantKnowledgeBase',
          func=cov_qa.run,
          description="Use this tool to check for covenant details and thresholds."
  )
   
  financial_tool = Tool(
      name="FinancialChecker",
      func=financial_qa_chain.run,
      description="Use this tool to retrieve financial information about the company."
  )
   
  cov_agent = initialize_agent(
      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      prompt=chat_prompt,
      tools=[cov_knowledge_tool, financial_tool, math_tool],
      llm=llm,
      verbose=True,
      #max_iterations=3,
      #early_stopping_method='generate',
      memory=conversational_memory
  )
   
  query = "Check if company XYZ has breached any covenants based on the financial data provided,  start your response with either 'Yes' or 'No'. Be detailed in your response."
   
  # Run the agent using the tools
  result = cov_agent.run(query)
  st.write(result)
 
 






























with tabs[1]:
 os.environ["TAVILY_API_KEY"] = "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"

 llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key = api_key)
 search = TavilySearchAPIWrapper()
 tavily_tool = TavilySearchResults(api_wrapper=search)

 agent_chain = initialize_agent(
  [tavily_tool],
  llm,
  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  )

 query = st.text_input('Ask the bankruptcy status', value="")

 if query:
  st.write(agent_chain.run(
     query,
 ))






















































































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
 
