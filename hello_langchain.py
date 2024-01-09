from langchain_community.chat_models import ChatOpenAI
import dotenv
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()
llm = ChatOpenAI()

result = llm.invoke("Who is Nilanjan Raychaudhuri?")
print(result)

# output_parser = StrOutputParser()
# print(output_parser.invoke(result))
