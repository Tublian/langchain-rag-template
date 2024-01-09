import os
import bs4
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import dotenv

dotenv.load_dotenv()

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(
    class_=(
        "title",
        "session-time",
        "room",
        "speakers",
        "description",
        "track",
        "tags",
        "format",
        "level",
    )
)

loader = WebBaseLoader(
    web_paths=["https://codemash.org/schedule/"],
)
loader.requests_kwargs = {"verify": False}
loader.bs_kwargs = {"parse_only": bs4_strainer}
docs = loader.load()

print(docs)

# splitting documents to fix into the context window
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./codemash_db"
)
