from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

import dotenv

from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()


# loading from disk
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./codemash_db", embedding_function=embedding)

# similarity search capabilities of a vector store to facillitate retrieval
retriever = vectorstore.as_retriever()

print(retriever)

# prompt = hub.pull("rlm/rag-prompt")

chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a codemash conference bot.
               You have knowledge of all the sessions.
               You know who, when and where each session is scheduled.
               You also know the details about each session (topic, speaker and time)""",
        ),
        (
            "human",
            """You are an assistant for question-answering tasks.
              Use the following pieces of retrieved context to answer the question.
              If you don't know the answer, just say that you don't know.
              Use five sentences maximum and keep the answer concise.
               Question: {question}
               Context: {context}
               Answer:""",
        ),
    ]
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    print("\n\n".join(doc.page_content for doc in docs))
    return "\n\n".join(doc.page_content for doc in docs)


# LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
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
