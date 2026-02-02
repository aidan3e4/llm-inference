import logging

from llm.inference import ModelConfig, InferenceConfig

logger = logging.getLogger(__name__)


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
