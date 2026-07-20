"""MCP client helpers for agents using LiteLLM."""

import json
from copy import deepcopy
from typing import Any

from loguru import logger
from mcp.types import ContentBlock, ImageContent, TextContent
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.models import LitellmInputMessage


IMAGE_ONLY_TOOL_BASENAMES = {
    "read_image",
    "read_image_file",
    "read_page_as_image",
}

IMAGE_ACTIONS_BY_META_TOOL = {
    "docs": {"read_image"},
    "pdf": {"page_as_image", "read_image"},
    "slides": {"read_image"},
}

TEXT_ONLY_TOOL_ERROR = (
    "Error: This image-returning tool/action is disabled because the configured "
    "model does not support image inputs. Use text extraction or metadata actions "
    "instead."
)


def _matches_tool_basename(name: str, basename: str) -> bool:
    """Match both unprefixed tools and gateway names such as pdf_server_pdf."""
    return name == basename or name.endswith(f"_{basename}")


def _blocked_image_actions(name: str) -> set[str]:
    for tool_basename, blocked_actions in IMAGE_ACTIONS_BY_META_TOOL.items():
        if name == tool_basename or name.endswith(f"_server_{tool_basename}"):
            return blocked_actions
    return set()


def _is_image_only_tool(name: str) -> bool:
    return any(
        _matches_tool_basename(name, basename)
        for basename in IMAGE_ONLY_TOOL_BASENAMES
    )


def _remove_enum_values(schema: Any, blocked_values: set[str]) -> None:
    """Remove blocked values from an action schema, including nested unions."""
    if isinstance(schema, dict):
        enum = schema.get("enum")
        if isinstance(enum, list):
            schema["enum"] = [value for value in enum if value not in blocked_values]
        for value in schema.values():
            _remove_enum_values(value, blocked_values)
    elif isinstance(schema, list):
        for value in schema:
            _remove_enum_values(value, blocked_values)


def filter_mcp_tools_for_model(
    tools: list[ChatCompletionToolParam], supports_vision: bool
) -> list[ChatCompletionToolParam]:
    """Hide image-only tools/actions from models configured as text-only."""
    if supports_vision:
        return tools

    filtered: list[ChatCompletionToolParam] = []
    for tool in tools:
        function = tool.get("function", {})
        name = str(function.get("name", ""))
        if _is_image_only_tool(name):
            continue

        blocked_actions = _blocked_image_actions(name)
        if not blocked_actions:
            filtered.append(tool)
            continue

        filtered_tool = deepcopy(tool)
        filtered_function = filtered_tool.get("function", {})
        parameters = filtered_function.get("parameters", {})
        if isinstance(parameters, dict):
            properties = parameters.get("properties", {})
            if isinstance(properties, dict) and "action" in properties:
                _remove_enum_values(properties["action"], blocked_actions)

        disabled = ", ".join(sorted(blocked_actions))
        description = str(filtered_function.get("description", "")).rstrip()
        filtered_function["description"] = (
            f"{description}\n\nText-only mode: the following image-returning "
            f"actions are disabled: {disabled}."
        ).strip()
        filtered.append(filtered_tool)

    return filtered


def get_multimodal_tool_call_block_reason(
    name: str, arguments: str, supports_vision: bool
) -> str | None:
    """Return an explanatory error when a text-only model requests image output."""
    if supports_vision:
        return None
    if _is_image_only_tool(name):
        return TEXT_ONLY_TOOL_ERROR

    blocked_actions = _blocked_image_actions(name)
    if not blocked_actions:
        return None

    try:
        parsed = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict) and parsed.get("action") in blocked_actions:
        return TEXT_ONLY_TOOL_ERROR
    return None


def build_mcp_gateway_schema(
    mcp_gateway_url: str,
    mcp_gateway_auth_token: str | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Build the MCP client config schema for connecting to the environment's MCP gateway.

    The gateway is a single HTTP endpoint that proxies to all configured MCP servers
    in the environment sandbox.

    Args:
        mcp_gateway_url: URL of the MCP gateway (e.g. "http://localhost:8000/mcp/")
        mcp_gateway_auth_token: Bearer token for authentication (None for local/unauthenticated)

    Returns:
        The standard schema expected by the MCP client.
    """
    gateway_config: dict[str, Any] = {
        "transport": "streamable-http",
        "url": mcp_gateway_url,
    }

    # Only add Authorization header if token is provided
    if mcp_gateway_auth_token:
        gateway_config["headers"] = {
            "Authorization": f"Bearer {mcp_gateway_auth_token}"
        }

    return {
        "mcpServers": {
            "gateway": gateway_config,
        }
    }


def content_blocks_to_messages(
    content_blocks: list[ContentBlock],
    tool_call_id: str,
    name: str,
    model: str,
    deferred_image_messages: list[LitellmInputMessage],
    supports_vision: bool = True,
) -> list[LitellmInputMessage]:
    """
    Convert MCP content blocks to a single LiteLLM tool message.

    Each tool_use must have exactly one tool_result. This function combines all
    content blocks into a single tool message to satisfy API requirements for
    Anthropic, OpenAI, and other providers.

    For non-Anthropic models, images cannot be embedded in tool results, so they
    are appended to deferred_image_messages as user messages. The caller is
    responsible for adding them to self.messages after all tool responses.
    This list is mutated in place.

    Args:
        content_blocks: MCP content blocks from tool result
        tool_call_id: The tool call ID to associate with the result
        name: The tool name
        model: The model being used
        deferred_image_messages: Mutable list that image user messages are
            appended to (mutated in place). Callers should extend self.messages
            with this list after all tool responses are added.
        supports_vision: Whether the configured model accepts image inputs. When
            false, image blocks are replaced with an explanatory text result.

    Returns:
        List containing exactly one tool message.
    """
    # Anthropic supports images directly in tool results
    supports_image_tool_results = supports_vision and model.startswith("anthropic/")

    text_contents: list[str] = []
    image_data_uris: list[str] = []
    omitted_image_count = 0

    for content_block in content_blocks:
        match content_block:
            case TextContent():
                block = TextContent.model_validate(content_block)
                text_contents.append(block.text)

            case ImageContent():
                block = ImageContent.model_validate(content_block)
                if supports_vision:
                    data_uri = f"data:{block.mimeType};base64,{block.data}"
                    image_data_uris.append(data_uri)
                else:
                    omitted_image_count += 1

            case _:
                logger.warning(f"Content block type {content_block.type} not supported")
                text_contents.append("Unable to parse tool call response")

    if omitted_image_count:
        text_contents.append(
            f"{omitted_image_count} image output(s) omitted because the configured "
            "model does not support image inputs. Use text extraction or metadata "
            "actions instead."
        )

    messages: list[LitellmInputMessage] = []

    if supports_image_tool_results:
        content: list[dict[str, Any]] = []
        for text in text_contents:
            content.append({"type": "text", "text": text})
        for data_uri in image_data_uris:
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        tool_message: LitellmInputMessage = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }  # pyright: ignore[reportAssignmentType]
        messages.append(tool_message)
    else:
        content = [{"type": "text", "text": text} for text in text_contents]

        if image_data_uris and not content:
            content.append(
                {"type": "text", "text": f"Image(s) returned by {name} tool"}
            )

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }  # pyright: ignore[reportAssignmentType]
        messages.append(tool_message)

        # Image workaround: non-Anthropic models don't support images in tool results,
        # so we append them to deferred_image_messages for the caller to add after all tool responses.
        for data_uri in image_data_uris:
            deferred_image_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            )

    return messages
