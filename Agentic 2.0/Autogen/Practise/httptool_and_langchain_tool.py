import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_agentchat.ui import Console
import os
from autogen_ext.tools.http import HttpTool


from autogen_ext.tools.http import HttpTool
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set. Please check env variable")


model_client=OpenAIChatCompletionClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="openai/gpt-4o-mini",
    model_info={
        "family":"openai",
        "json_output":False,
        "structured_output":True,
        "vision": False,
        "function_calling": True
    }
)

schema = {
    "type": "object",
    "properties": {
        "isbn": {"type": "string", "description": "ISBN number"},
    },
    "required": ["isbn"]
}

# Use full query param in path with curly braces
http_tool = HttpTool(
    name="HttpTool",
    description="Get book details by ISBN",
    host="demoqa.com",
    port=443,
    scheme="https",
    path="/BookStore/v1/Book?isbn={isbn}",  # ← force query param in path
    method="GET",
    json_schema=schema,
    return_type="json"
)

tool2 = HttpTool(
    name="get_ip",
    description="Get your IP address",
    scheme="https",
    host="httpbin.org",
    port=443,
    path="/ip",
    method="GET",
    json_schema={"type": "object", "properties": {}},
    return_type="json"
)


httptool_agent=AssistantAgent(
    name="httptool_agent",
    model_client=model_client,
    description="HTTP Tool exexution agent",
    system_message="You are an helpful assistant to execute any API call using http tool",
    tools=[http_tool]
)
async def main():
    await Console(httptool_agent.run_stream(task="Get book details for isbn=9781449325862"))

if(__name__=="__main__"):
    asyncio.run(main())

