import json
import logging

from llm.inference import ModelConfig, litellm_call, save_messages
from llm.tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)


async def agentic_turn(
    messages: list, tools: list, config: ModelConfig = None
) -> str | None:
    """
    An agentic turn just means an AI turn + tool execution.

    Returns the final answer if done, None if another turn is needed.
    """
    response = await litellm_call(messages, config=config, tools=tools)

    message = response.choices[0].message
    messages.append(message.dict(exclude_none=True))  # Add assistant response

    if not message.tool_calls:
        print("Final answer:", message.content)
        return message.content

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

    return None  # Continue the loop


async def agentic_session(
    messages: list,
    tools: list = None,
    config: ModelConfig = None,
    do_save_messages: bool = True,
) -> str:
    """An agentic session here just means an agentic turn"""
    while True:
        answer = await agentic_turn(messages, tools, config)
        if answer is not None:
            break

    if do_save_messages:
        save_messages(messages)
