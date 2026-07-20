from typing import cast

from mcp.types import ImageContent
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.utils.llm import _is_non_retriable_bad_request, _should_skip_retry
from runner.utils.mcp import (
    content_blocks_to_messages,
    filter_mcp_tools_for_model,
    get_multimodal_tool_call_block_reason,
)


def _tool(
    name: str, actions: list[str] | None = None
) -> ChatCompletionToolParam:
    parameters: dict = {"type": "object", "properties": {}}
    if actions is not None:
        parameters["properties"]["action"] = {
            "type": "string",
            "enum": actions,
        }
    return cast(
        ChatCompletionToolParam,
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool {name}",
                "parameters": parameters,
            },
        },
    )


def test_multimodal_bad_request_is_non_retriable() -> None:
    error = Exception("ValueError: /model is not a multimodal model")
    assert _is_non_retriable_bad_request(error)
    assert _should_skip_retry(error)


def test_text_only_tool_filter_removes_image_tools_and_actions() -> None:
    tools = [
        _tool("filesystem_server_read_image_file"),
        _tool(
            "pdf_server_pdf",
            ["help", "read_pages", "read_image", "page_as_image", "search"],
        ),
        _tool("filesystem_server_read_text_file"),
    ]

    filtered = filter_mcp_tools_for_model(tools, supports_vision=False)
    by_name = {tool["function"]["name"]: tool for tool in filtered}

    assert "filesystem_server_read_image_file" not in by_name
    assert "filesystem_server_read_text_file" in by_name
    pdf_actions = by_name["pdf_server_pdf"]["function"]["parameters"][
        "properties"
    ]["action"]["enum"]
    assert pdf_actions == ["help", "read_pages", "search"]


def test_text_only_execution_guard_blocks_hallucinated_image_action() -> None:
    assert get_multimodal_tool_call_block_reason(
        "pdf_server_pdf",
        '{"action":"page_as_image","file_path":"/report.pdf","page_number":1}',
        supports_vision=False,
    )
    assert not get_multimodal_tool_call_block_reason(
        "pdf_server_pdf",
        '{"action":"read_pages","file_path":"/report.pdf"}',
        supports_vision=False,
    )


def test_text_only_result_conversion_never_emits_image_input() -> None:
    deferred_messages = []
    messages = content_blocks_to_messages(
        [ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")],
        tool_call_id="call-1",
        name="unexpected_image_tool",
        model="openai/text-only-model",
        deferred_image_messages=deferred_messages,
        supports_vision=False,
    )

    assert deferred_messages == []
    assert "image output(s) omitted" in str(messages[0]["content"])
    assert "image_url" not in str(messages)
