from dotenv import load_dotenv
import logging

from logger import setup_logging
from llm.orchestration import agentic_session
from llm.tools import TOOLS


load_dotenv()

setup_logging(logging.DEBUG)

if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that responds to the user.",
        },
        {
            "role": "user",
            "content": "what's the weather in Lausanne Switzerland today ?",
        },
    ]
    tools = TOOLS

    agentic_session(messages=messages, tools=tools)
