import getpass
import os
import asyncio

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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
async def call_model(state: MessagesState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

async def main():
    query = "Hi! I'm Bob."
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

    # Switch user, and it will start over
    config2 = {"configurable": {"thread_id": "abc234"}}
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config2)
    output["messages"][-1].pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
