import os
import streamlit as st
import tiktoken
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, ElasticsearchStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import shutil

load_dotenv(find_dotenv(), override=True)

chunk_size = 1000
chunk_overlap = 50
k=3

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible.

Use only given context to answer the question - do not answer from the context that's not been provided to you. 

{context}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# Elasticsearch Vector store
LOAN_DOCS_INDEX = "loans-docs"

CLOUD_ID = os.getenv("CLOUD_ID")
ES_API_KEY = os.getenv("ES_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

es_vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_api_key=ES_API_KEY,
    index_name=LOAN_DOCS_INDEX,
    embedding=embeddings
)

# Start with a clean state

def cleanup_chromadb():
    chromadb_dir = os.path.join(os.getcwd(),"chromadb")

    if os.path.exists(chromadb_dir):
        print("removing Chromadb dir")
        shutil.rmtree(chromadb_dir)
    else:
        print("chromadb doesn't exist")


def cleanup_uploads():
    dir_path = os.path.join(os.getcwd(),"uploads")

    # Iterate over all items in the directory
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)

        # Remove the item (file or directory)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Remove files and links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directories
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


# Function to help loading PDF and DOCX files
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Function to chunkify data
def chunk_data(data):
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)
    
    chunks = rec_splitter.split_documents(data)
    print("Chunking completed")

    return chunks

# Embeddings store to local ChromaDB
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory='./chromadb')

    return vector_store

# Create embeddings using OpenAIEmbeddings()
# And save the Embeddings in Elasticsearch vector store
def create_embeddings_elasticsearch(chunks):
    
    embeddings = OpenAIEmbeddings()

    vector_store = es_vector_store.from_documents(
        chunks, 
        embeddings, 
        index_name=LOAN_DOCS_INDEX, 
        es_cloud_id=CLOUD_ID, 
        es_api_key=ES_API_KEY)

    print("Embeddings persisted to Elasticsearch Vector Store")

    return vector_store

def answer(vector_store, q, k=3):
    
    # Instantiate appropriate model
    llm = ChatOpenAI(model=MODEL, temperature=0.8)

    retriever = vector_store.as_retriever(
        search_type='similarity', 
        search_kwargs={'k': k})
    
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": QA_PROMPT})

    answer = chain.run(q)
    return answer

# Funtion to calculate embedding cost
def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    
    return total_tokens, total_tokens / 1000 * 0.0004


# Purge the chat history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Purge Elasticsearch Index (not working)
def clear_elastic_index():
    es_vector_store.delete()

# Main
if __name__ == "__main__":
    cleanup_chromadb()
    cleanup_uploads()

    # api_key = st.text_input('OpenAI API Key:', type='password')
    # if api_key:
    #     os.environ['OPENAI_API_KEY'] = api_key
    
    uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx'])
    # add data button widget
    add_data = st.button('Upload File', on_click=clear_history)

    if uploaded_file and add_data: 
        # clear_elastic_index()
        
        with st.spinner('Processing the file..'):
            # clear_elastic_index()

            bytes_data = uploaded_file.read()
            file_name = os.path.join('./uploads', uploaded_file.name)
            with open(file_name, 'wb') as f:
                f.write(bytes_data)

            data = load_document(file_name)
            chunks = chunk_data(data)

            tokens, embedding_cost = calculate_embedding_cost(chunks)
            st.write(f'Embedding cost: ${embedding_cost:.4f}')

            # print("using Elasticsearch vector store")
            # vector_store = create_embeddings_elasticsearch(chunks)
            print("using Chroma local vector store")
            vector_store = create_embeddings(chunks)

            st.session_state.vs = vector_store
            st.success('File uploaded successfully. You are ready to ask your question!')

    # user's question text input widget
    question = st.text_input('Ask a question:')
    if question:
        if 'vs' in st.session_state: 
            vector_store = st.session_state.vs
            # st.write(f'k: {k}')
            answer = answer(vector_store, question, k)
            
            st.text_area("Here's your answer:", value=answer)