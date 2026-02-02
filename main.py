import argparse
import asyncio
import logging

from logger import setup_logging
from llm.inference import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    ModelConfig,
    InferenceConfig,
    llm_turn,
)
from llm.tools import TOOLS

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

    setup_logging(logging.DEBUG)

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.message},
    ]

    model_config = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    inference_config = InferenceConfig()

    answer = asyncio.run(
        llm_turn(
            messages=messages,
            tools=TOOLS,
            model_config=model_config,
            inference_config=inference_config,
        )
    )
    print(answer)
