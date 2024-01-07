import os
import bs4
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import dotenv

dotenv.load_dotenv()

loader = WebBaseLoader(
    web_paths=["https://codemash.org/schedule"],
)
loader.requests_kwargs = {"verify": False}
docs = loader.load()

print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./codemash_db"
)
