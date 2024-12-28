import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from dotenv import load_dotenv
import bs4
from bs4 import SoupStrainer

#from langsmith import LangSmith
load_dotenv()
#langsmith = LangSmith(
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="pr-kindly-cutlet-35"
    #api_key=os.getenv("LANGCHAIN_API_KEY", "your-langsmith-api-key"),
    #endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
   # project=os.getenv("LANGCHAIN_PROJECT", "your-project-name"),
   # tracing_v2=LANGCHAIN_TRACING_V2  # Should be a Python boolean (True/False)

# Load environment variables
#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
#LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
#LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
#LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
#LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
# Document loader
@st.cache_data
def load_document_loader():
    loader = WebBaseLoader(
    'https://en.wikipedia.org/wiki/Elon_Musk',
    bs_kwargs=dict(parse_only=SoupStrainer(class_=('mw-content-ltr mw-parser-output')))
    )
    documents = loader.load()
    # Split documents into chunks
    recursive = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = recursive.split_documents(documents)
    return chunks 
chunks=load_document_loader()
# Initialize embedding and Qdrant
embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Qdrant setup
api_key = os.getenv('qdrant_api_key')
url = 'https://1328bf7c-9693-4c14-a04c-f342030f3b52.us-east4-0.gcp.cloud.qdrant.io:6333'
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    url=url,
    api_key=api_key,
    prefer_grpc=True,
    collection_name="Elon Muske"
)

# Initialize Google LLM
google_api = os.getenv('google_api_key')
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

# Setup retriever and chain
num_chunks = 5
retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements.
Context: {context}
Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Chain setup
query_fetcher = itemgetter("question")
setup = {"question": query_fetcher, "context": query_fetcher | retriever }
_chain = setup | _prompt | llm | StrOutputParser()

# Streamlit UI
# Streamlit UI
st.title("Ask Anything About Elon Musk")

# Chat container to display conversation
chat_container = st.container()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
def clear_input_field():
    if st.session_state.user_question == "":
        st.session_state.user_question = st.session_state.user_input
        st.session_state.user_input = ""
        
def send_input():
    st.session_state.send_input=True
    clear_input_field()

# Input field for queries
with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn", on_click=clear_input_field)  # Single send button

# Chat logic
if send_button or st.session_state.send_input:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response = _chain.invoke({'question': query})  # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
