import argparse
import logging

from dotenv import load_dotenv

from logger import setup_logging
from llm.inference import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    ModelConfig,
)
from llm.orchestration import agentic_session
from llm.tools import TOOLS


load_dotenv()


def main(
    user_message: str,
    system_message: str = "You are a helpful assistant that responds to the user.",
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tools: list = None,
    save_messages: bool = True,
) -> str:
    """Run an agentic session with the given configuration.

    Args:
        user_message: The user's input message (can be a string or a list for multimodal content).
        system_message: The system prompt.
        model_name: The model to use.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        tools: List of tool definitions. Defaults to TOOLS if None.

    Returns:
        The final answer from the agent.
    """
    setup_logging(logging.DEBUG)

    if tools is None:
        tools = TOOLS

    config = ModelConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return agentic_session(
        messages=messages, tools=tools, config=config, save_messages=save_messages
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an agentic LLM session")
    parser.add_argument(
        "message",
        nargs="?",
        default="what's the weather in Lausanne Switzerland today ?",
        help="User message to send to the agent",
    )
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens"
    )
    parser.add_argument(
        "--system",
        "-s",
        default="You are a helpful assistant that responds to the user.",
        help="System message",
    )

    args = parser.parse_args()

    answer = main(
        user_message=args.message,
        system_message=args.system,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"\nReturned answer: {answer}")
