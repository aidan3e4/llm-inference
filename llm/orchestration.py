from dotenv import load_dotenv
import json
import logging

from llm.inference import litellm_call, save_messages
from logger import logger, setup_logging
from llm.tools import TOOL_FUNCTIONS

load_dotenv()

setup_logging(logging.DEBUG)


def agentic_turn(messages: list, tools: list) -> bool:
    """Returns whether we do another turn or no"""
    response = litellm_call(messages, tools=tools)

    message = response.choices[0].message
    messages.append(message.dict(exclude_none=True))  # Add assistant response

    if not message.tool_calls:
        print("Final answer:", message.content)
        return False

    # Execute the tool call(s)
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        if func_name in TOOL_FUNCTIONS:
            args = json.loads(tool_call.function.arguments)
            result = TOOL_FUNCTIONS[func_name](**args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result,
                }
            )
        else:
            logger.warning(f"Unknown tool: {func_name}")

    return True

    # Loop continues → model gets results and decides next step or final answer


def agentic_session(messages: list, tools=None):
    should_continue = True
    while should_continue:
        should_continue = agentic_turn(messages, tools)

    save_messages(messages)
