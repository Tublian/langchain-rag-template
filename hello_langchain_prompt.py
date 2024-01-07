from langchain_community.chat_models import ChatOpenAI
import dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are world class developer and you know all the great software developers in the world.",
        ),
        ("user", "{input}"),
    ]
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# result1 = chain.invoke({"input": "Who is Nilanjan Raychaudhuri?"})
# print(result1)


def main():
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result1 = chain.invoke({"input": user_input})
        print(result1)


if __name__ == "__main__":
    main()
