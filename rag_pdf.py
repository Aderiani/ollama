
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

doc_path = "./data/sample.pdf"

model = "kimi-k2.5:cloud"

if doc_path:
    loader = UnstructuredPDFLoader(doc_path)
    data = loader.load()
    print(f"Loaded {len(data)} pages from local PDF.")
else:
    print("No local PDF path provided. Please provide a valid path.")
    
content = "\n".join([page.page_content for page in data])
print(content[:500])  # Print first 500 characters of the content

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)
print(f"Split into {len(texts)} chunks.")