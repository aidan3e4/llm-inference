"""Tool definitions and implementations"""

import inspect
import logging
from typing import Annotated, Callable, get_args, get_origin, get_type_hints

from ddgs import DDGS

logger = logging.getLogger(__name__)

# Registry of tools
TOOLS: list[dict] = []
TOOL_FUNCTIONS: dict[str, Callable] = {}

# Type mapping for JSON schema
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(func: Callable) -> Callable:
    """
    Decorator that registers a function as a tool.

    Uses the function's docstring as description and type hints for schema.
    Use Annotated[type, "description"] for parameter descriptions.

    Example:
        @tool
        def search(query: Annotated[str, "The search query"]) -> str:
            '''Search the web for information.'''
            ...
    """
    hints = get_type_hints(func, include_extras=True)
    sig = inspect.signature(func)

    # Extract description from docstring (first line)
    description = (func.__doc__ or "").strip().split("\n")[0]

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        hint = hints.get(param_name, str)

        # Handle Annotated[type, description]
        param_description = None
        param_type = hint
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            param_type = args[0]
            if len(args) > 1 and isinstance(args[1], str):
                param_description = args[1]

        # Build property schema
        prop = {"type": TYPE_MAP.get(param_type, "string")}
        if param_description:
            prop["description"] = param_description

        properties[param_name] = prop

        # Required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    # Build tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    TOOLS.append(tool_schema)
    TOOL_FUNCTIONS[func.__name__] = func

    return func


# --- Tool implementations ---


@tool
def web_search(
    query: Annotated[str, "The search query"],
    max_results: Annotated[int, "Maximum number of results"] = 5,
) -> str:
    """Search the web for current information."""
    logger.debug(f"Searching DuckDuckGo for: {query}")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    logger.debug(f"DuckDuckGo returned {len(results)} results")

    search_content = "\n".join(
        [f"- {r['title']}: {r['body']} ({r['href']})" for r in results]
    )

    return f"Search results for '{query}':\n{search_content}"
