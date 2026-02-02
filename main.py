from llm.orchestration import agentic_session
from llm.tools import TOOLS


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

    agentic_session(messages=messages, tools=TOOLS)
