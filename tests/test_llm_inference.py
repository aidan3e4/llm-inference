from dotenv import load_dotenv
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

load_dotenv()

from llm_inference.llm.inference import litellm_call, ModelConfig
import pytest

@pytest.mark.asyncio
async def test_inference_with_max_tokens_too_large():
    model_config = ModelConfig(model_name="gpt-4o", max_tokens=1_000_000)
    result = await litellm_call(messages=[{"role": "user", "content":"Just say YES"}], config=model_config)
    print("done")

