import getpass
import os

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env", override=True)
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.messages import HumanMessage

response1 = llm.invoke([HumanMessage(content="Hi! I'm Bob")]).content
print(response1)

print('*' * 40)
response2 = llm.invoke([HumanMessage(content="What is my name?")])
print(response2.content)

print('*' * 40)
from langchain_core.messages import AIMessage
response3 = llm.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content=response1),
        HumanMessage(content="What's my name?"),
    ]
)
print(response3.content)
