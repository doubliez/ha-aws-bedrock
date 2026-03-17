"""Base entity for AWS Bedrock Conversation integration."""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from botocore.exceptions import ClientError
from voluptuous_openapi import convert

from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from . import BedrockConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
    TOOL_USE_SUPPORTED_MODELS,
)

_LOGGER = logging.getLogger(__name__)


def _model_supports_tools(model_id: str) -> bool:
    """Return True if the model supports tool use via the Converse API."""
    return any(model_id.startswith(prefix) for prefix in TOOL_USE_SUPPORTED_MODELS)


def _format_tool(tool: llm.Tool) -> dict[str, Any]:
    """Convert a HA LLM Tool to Bedrock toolSpec format."""
    schema: dict[str, Any] = convert(tool.parameters) if tool.parameters else {}
    # Bedrock requires the schema to have an explicit object type and properties
    if "type" not in schema:
        schema["type"] = "object"
    if "properties" not in schema:
        schema["properties"] = {}

    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description or f"Tool: {tool.name}",
            "inputSchema": {"json": schema},
        }
    }


def _convert_messages(
    content_list: list,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert HA ChatLog content to Bedrock Converse API format.

    Returns (system, messages) where:
    - system is a list of system prompt blocks
    - messages is a list of user/assistant turn dicts
    """
    system: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []

    for item in content_list:
        if isinstance(item, SystemContent):
            if item.content:
                system = [{"text": item.content}]

        elif isinstance(item, UserContent):
            messages.append(
                {
                    "role": "user",
                    "content": [{"text": item.content}],
                }
            )

        elif isinstance(item, AssistantContent):
            bedrock_content: list[dict[str, Any]] = []
            if item.content:
                bedrock_content.append({"text": item.content})
            if item.tool_calls:
                for tc in item.tool_calls:
                    bedrock_content.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.tool_name,
                                "input": tc.tool_args,
                            }
                        }
                    )
            if bedrock_content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": bedrock_content,
                    }
                )

        elif isinstance(item, ToolResultContent):
            # Tool results must be sent as user-role messages in Bedrock.
            # Multiple consecutive tool results (from a single assistant turn with
            # multiple tool calls) must be grouped into a single user message.
            tool_result_block: dict[str, Any] = {
                "toolResult": {
                    "toolUseId": item.tool_call_id,
                    "content": [{"text": json.dumps(item.tool_result)}],
                    "status": "success",
                }
            }
            if (
                messages
                and messages[-1]["role"] == "user"
                and messages[-1]["content"]
                and "toolResult" in messages[-1]["content"][0]
            ):
                # Append to the existing tool-result user message
                messages[-1]["content"].append(tool_result_block)
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [tool_result_block],
                    }
                )

    return system, messages


class BedrockBaseLLMEntity(Entity):
    """Base entity for AWS Bedrock LLM interactions."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self, entry: BedrockConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Amazon Web Services",
            model="Amazon Bedrock",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _client(self) -> Any:
        """Return the Bedrock runtime client from the config entry."""
        return self.entry.runtime_data

    async def _async_handle_chat_log(self, chat_log: ChatLog) -> None:
        """Generate a response using the AWS Bedrock Converse API.

        Implements an agentic tool-use loop: the model can call Home Assistant
        tools (entity control, service calls, etc.) multiple times before
        producing a final text response.
        """
        options = self.subentry.data
        model_id: str = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens: int = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature: float = options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        supports_tools = _model_supports_tools(model_id)

        response: dict[str, Any] = {}

        for iteration in range(MAX_TOOL_ITERATIONS):
            system, messages = _convert_messages(chat_log.content)

            # Build tool config when the model supports tools and HA has tools available
            tool_config: dict[str, Any] | None = None
            if supports_tools and chat_log.llm_api and chat_log.llm_api.tools:
                tool_config = {
                    "tools": [_format_tool(t) for t in chat_log.llm_api.tools],
                    "toolChoice": {"auto": {}},
                }

            converse_kwargs: dict[str, Any] = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            }
            if system:
                converse_kwargs["system"] = system
            if tool_config:
                converse_kwargs["toolConfig"] = tool_config

            try:
                response = await self.hass.async_add_executor_job(
                    functools.partial(self._client.converse, **converse_kwargs)
                )
            except ClientError as err:
                error_code = err.response.get("Error", {}).get("Code", "")
                error_message = err.response.get("Error", {}).get(
                    "Message", str(err)
                )
                _LOGGER.error(
                    "AWS Bedrock API error [%s]: %s", error_code, error_message
                )
                raise HomeAssistantError(
                    f"AWS Bedrock error ({error_code}): {error_message}"
                ) from err

            stop_reason: str = response.get("stopReason", "end_turn")
            response_message = response["output"]["message"]

            # Parse text content and tool calls from the response
            text_parts: list[str] = []
            tool_calls: list[llm.ToolInput] = []

            for block in response_message.get("content", []):
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_calls.append(
                        llm.ToolInput(
                            id=tool_use["toolUseId"],
                            tool_name=tool_use["name"],
                            tool_args=tool_use.get("input", {}),
                        )
                    )

            # Add the assistant response to the chat log.
            # async_add_assistant_content executes internal tool calls and
            # automatically appends ToolResultContent entries to the log.
            assistant_content = AssistantContent(
                agent_id=self.entity_id,
                content="".join(text_parts) or None,
                tool_calls=tool_calls or None,
            )
            async for _ in chat_log.async_add_assistant_content(assistant_content):
                pass

            # Stop looping if the model gave a final answer or made no tool calls
            if stop_reason != "tool_use" or not tool_calls:
                break

            _LOGGER.debug(
                "Tool-use loop iteration %d/%d (model=%s)",
                iteration + 1,
                MAX_TOOL_ITERATIONS,
                model_id,
            )

        if usage := response.get("usage"):
            _LOGGER.debug(
                "Token usage — input: %s, output: %s",
                usage.get("inputTokens"),
                usage.get("outputTokens"),
            )
