import json
import logging
from dataclasses import dataclass

from llm.inference import ModelConfig, litellm_call
from llm.tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)


DEFAULT_MAX_TURNS_LLM_CONSECUTIVE = 5
DEFAULT_MAX_TURNS_SESSION = 20


@dataclass
class InferenceConfig:
    max_turns_llm_consecutive: int = DEFAULT_MAX_TURNS_LLM_CONSECUTIVE
    max_turns_session: int = DEFAULT_MAX_TURNS_SESSION


async def llm_turn(
    messages: list,
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    tools: list = None,
) -> str:
    """
    Run an llm turn that continues calling the LLM and executing tools
    until the LLM responds without tool calls or max turns is reached.
    """
    turn_count = 0

    while (
        turn_count < inference_config.max_turns_llm_consecutive
        and len(messages) < inference_config.max_turns_session
    ):
        turn_count += 1
        logger.info(
            f"LLM consecutive turn {turn_count}/{inference_config.max_turns_llm_consecutive}"
        )

        response = await litellm_call(messages, config=model_config, tools=tools)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content

        # Execute tool calls
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": f"Error: Unknown tool '{func_name}'",
                    }
                )

    logger.warning(
        f"Max LLM consecutive turns ({inference_config.max_turns_llm_consecutive}) reached, stopping turn"
    )

    return messages[-1].get("content") if isinstance(messages[-1], dict) else None


async def llm_session(
    messages: list,
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    tools: list = None,
    do_save_messages: bool = True,
):
    """To implement as needed, involves more than just an LLM turn"""
    pass
    # if do_save_messages:
    #     save_messages(messages)
