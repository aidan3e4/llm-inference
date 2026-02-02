import json
import logging
from datetime import datetime

from litellm import completion

from constants import data_dir

logger = logging.getLogger(__name__)


MODEL_NAME = "novita/moonshotai/kimi-k2.5"
MAX_TOKENS = 262144
TEMPERATURE = 0.7


def litellm_call(
    messages: list,
    model_name: str = MODEL_NAME,
    tools: list = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
):
    logger.info("Starting the LLM call")
    response = completion(
        model=model_name,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logger.info("LLM call succeeded")
    return response


def save_messages(messages: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_dir / f"{timestamp}.json"
    output_path.write_text(json.dumps(messages, indent=2))
    logger.info(f"Conversation saved to {output_path}")
