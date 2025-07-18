from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from dotenv import load_dotenv
from autogen_core.tools import FunctionTool
import os
import asyncio
from autogen_agentchat.ui import Console
from pydantic import BaseModel


from markdown import Markdown

load_dotenv();

api_key= os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER API key missing in env variables. Please check")

class WeatherInfo(BaseModel):
    city:str
    degrees:str

model_client= OpenAIChatCompletionClient(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model_info={
        "family":"deepseek",
        "function_calling":True,
        "json_output":True,
        "structured_output":True,
        "vision":False
    }
)


async def reverse_string(input: str):
    """
    This tools is used to reverse the string
    """
    print("reverser string",input[::-1])

reverse_string_tool = FunctionTool(reverse_string,description="This tools is used to reverse the string")

# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

get_weather_tool = FunctionTool(get_weather,description="This tools is used to get weather",strict=True)
print(get_weather_tool.schema)

agent= AssistantAgent(
    name="my_assistant_agent",
    model_client=model_client,
    system_message="You are an helpful AI assisytnat",
    tools=[get_weather_tool],
    output_content_type=WeatherInfo
    )

async def main():
    #response=await model_client.create([UserMessage(content="Can you reverse the string ""Hello\"",source="User")],tools=[reverse_string_tool],)
    #print(response.content)
    response= await agent.run(task="What is the weather in newyork city")
    print(response)


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent_weather = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


# # Run the agent and stream the messages to the console.
# async def main() -> None:
#     await Console(agent_weather.run_stream(task="What is the weather in New York?"))
#     # Close the connection to the model client.
#     await model_client.close()  

if(__name__=="__main__"):
    asyncio.run(main()) 
 
