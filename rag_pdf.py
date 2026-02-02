
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import ollama
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

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

ollama.pull(model="nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=texts,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="pdf_collection"
)
print("Vector store created with Ollama embeddings.")

llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)


retriver = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
    prompt= QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriver, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is the main topic discussed in the PDF document?"
response = chain.invoke({"question": question})
print("Response:")
print(response)
