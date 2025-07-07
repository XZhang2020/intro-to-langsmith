import getpass
import os

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=True)
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


from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)


llm = ChatOpenAI(
    temperature=0,
    model="qwen-plus",
    base_url=os.getenv("QWEN_BASE_URL"),
    api_key=os.getenv("QWEN_API_KEY")
)

# 定义消息列表
messages = [
    SystemMessage("Translate the following from English into Dutch"),
    HumanMessage("hi, how are you doing today?"),
]
# 调用 llm 对象的 invoke 方法，传入消息列表并打印响应内容
response = llm.invoke(messages)
print(type(response), '\n', response)
# print('*' * 40)
# print(response.content)
#
# for token in llm.stream(messages):
#     print(token.content, end="|")

