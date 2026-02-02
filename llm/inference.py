from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import json
import logging

from litellm import acompletion

from constants import data_dir

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL_NAME = "novita/moonshotai/kimi-k2.5"
DEFAULT_MAX_TOKENS = 262144
DEFAULT_TEMPERATURE = 0.7


@dataclass
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


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


def save_messages(messages: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"{timestamp}.json"
    output_path.write_text(json.dumps(messages, indent=2))
    logger.info(f"Conversation saved to {output_path}")
