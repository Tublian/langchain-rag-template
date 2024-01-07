from langchain_community.chat_models import ChatOpenAI
import dotenv

dotenv.load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI()

result = llm.invoke("Who is Nilanjan Raychaudhuri?")
print(result)
