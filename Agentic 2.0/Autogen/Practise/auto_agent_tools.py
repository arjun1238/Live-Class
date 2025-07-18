import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print("API key",api_key)
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set in the env file. Please check")

model_client= OpenAIChatCompletionClient(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model_info={
        "family":"openai",
        "json_output":False,
        "structured_output":False,
        "vision": False,
        "function_calling": True
    }
)

def reverse_string(text: str) -> str:
    '''
    Reverse the given text

    input:str

    output:str

    The reverse string is returned.
    '''
    return "Hello how are you?"

reverse_tool = FunctionTool(reverse_string,description='A tool to reverse a string')


my_agent1= AssistantAgent(
    name= "my_first_agent_in_py",
    model_client=model_client,
    system_message="You are an helpful assistant that reverse the string",
    tools=[reverse_string],
    reflect_on_tool_use= True
)

async def main():

    response= await my_agent1.run(task='Reverse the string "Hello, World!"')
    print(response)

if(__name__=="__main__"):
    asyncio.run(main())

