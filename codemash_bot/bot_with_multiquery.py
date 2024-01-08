from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

dotenv.load_dotenv()


# loading from disk
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./codemash_db", embedding_function=embedding)

retriever = vectorstore.as_retriever()

print(retriever)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


from langchain.retrievers.multi_query import MultiQueryRetriever

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)


def format_docs(docs):
    # print("\n\n".join(doc.page_content for doc in docs))
    return "\n\n".join(doc.page_content for doc in docs)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever_from_llm | format_docs
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

chat_history = []


def main():
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result1 = rag_chain.invoke(
            {"question": user_input, "chat_history": chat_history}
        )
        print(result1)
        chat_history.extend(
            [HumanMessage(content=user_input), AIMessage(content=result1)]
        )


if __name__ == "__main__":
    main()
