from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import json
import logging

from litellm import acompletion

from ..constants import data_dir
from .tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL_NAME = "novita/moonshotai/kimi-k2.5"
DEFAULT_MAX_TOKENS = 262144
DEFAULT_TEMPERATURE = 0.7

DEFAULT_MAX_TURNS_LLM_CONSECUTIVE = 5
DEFAULT_MAX_TURNS_SESSION = 20


@dataclass
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class InferenceConfig:
    max_turns_llm_consecutive: int = DEFAULT_MAX_TURNS_LLM_CONSECUTIVE
    max_turns_session: int = DEFAULT_MAX_TURNS_SESSION


async def litellm_call(
    messages: list,
    config: ModelConfig = None,
    tools: list = None,
):
    if config is None:
        config = ModelConfig()
    logger.info("Starting the LLM call")
    response = await acompletion(
        model=config.model_name,
        messages=messages,
        tools=tools,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    logger.info("LLM call succeeded")
    return response


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


def save_messages(messages: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"{timestamp}.json"
    output_path.write_text(json.dumps(messages, indent=2))
    logger.info(f"Conversation saved to {output_path}")
