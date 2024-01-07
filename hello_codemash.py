import os
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate


import dotenv

dotenv.load_dotenv()

loader = WebBaseLoader(
    web_paths=["https://codemash.org/schedule"],
)
loader.requests_kwargs = {"verify": False}
docs = loader.load()

print(docs)


# TODO
# separate store from retrieval
# feed in chat history along with the context


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()

print(retriever)

prompt = hub.pull("rlm/rag-prompt")

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a technical conference organizer of CodeMash conference. You know all about the sessions, speakers and time",
#         ),
#         (
#             "human",
#             """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#                   Question: {question}
#                  Context: {context}
#                  Answer:""",
#         )
#         # ("user", "{input}"),
#     ]
# )

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def main():
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result1 = rag_chain.invoke(user_input)
        print(result1)


if __name__ == "__main__":
    main()

# print("invoking...")
# result = rag_chain.invoke("How long is the Open Source internship?")
# print(result)
# print("invoking...1")
