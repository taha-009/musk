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

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
# Document loader
loader = WebBaseLoader(
    'https://en.wikipedia.org/wiki/Elon_Musk',
    bs_kwargs=dict(parse_only=SoupStrainer(class_=('mw-content-ltr mw-parser-output')))
)
documents = loader.load()
# Split documents into chunks
recursive = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = recursive.split_documents(documents)

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
# Initialize session state for input handling
if "query" not in st.session_state:
    st.session_state.query = ""  # Store the current query
if "send_flag" not in st.session_state:
    st.session_state.send_flag = False  # Track if the query should be sent

# Function to handle input changes
def handle_input_change():
    st.session_state.send_flag = True  # Set the flag to send query

# Input field for queries
with st.container():
    st.text_input(
        "Please enter a query",
        key="query",
        on_change=handle_input_change,  # Trigger this function when Enter is pressed
        label_visibility="collapsed",
    )
    send_button = st.button("Send", key="send_btn")  # Send button

# Chat logic (process query only when appropriate)
if (st.session_state.send_flag or send_button) and st.session_state.query.strip():
    with st.spinner("Processing... Please wait!"):
        response = _chain.invoke({'question': st.session_state.query})  # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", st.session_state.query))
    st.session_state.messages.append(("ai", response))
    # Reset flags and input field
    st.session_state.query = ""  # Clear the input field
    st.session_state.send_flag = False  # Reset the flag

# Display chat history from session state
chat_container = st.container()
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
