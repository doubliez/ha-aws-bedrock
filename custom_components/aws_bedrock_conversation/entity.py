"""Base entity for AWS Bedrock Conversation integration."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import AsyncIterator
from datetime import date, datetime
from typing import Any

from botocore.exceptions import ClientError
from voluptuous_openapi import convert

from homeassistant.components.conversation import (
    AssistantContent,
    AssistantContentDeltaDict,
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


class _ToolResultEncoder(json.JSONEncoder):
    """JSON encoder that handles types commonly found in HA tool results."""

    def default(self, o: object) -> object:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


def _model_supports_tools(model_id: str) -> bool:
    """Return True if the model supports tool use via the Converse API.

    Cross-region inference profiles (e.g. "us.amazon.nova-pro-v1:0") add a
    two-letter region prefix before the real model ID, so we strip it first.
    """
    base_id = model_id
    for region_prefix in ("us.", "eu.", "ap.", "us-gov."):
        if model_id.startswith(region_prefix):
            base_id = model_id[len(region_prefix):]
            break
    return any(base_id.startswith(prefix) for prefix in TOOL_USE_SUPPORTED_MODELS)


def _bedrock_tool_name(name: str) -> str:
    """Return a Bedrock-safe tool name.

    Bedrock only allows [a-zA-Z][a-zA-Z0-9_]* in tool names.  HA's merged-API
    namespacing produces names like "aws-bedrock-tools__get_dashboards" (hyphens
    from slugifying the API display name).  Replace every hyphen with underscore
    so Bedrock accepts them, and strip any other disallowed characters.
    """
    return name.replace("-", "_")


def _format_tool(tool: llm.Tool, bedrock_name: str) -> dict[str, Any]:
    """Convert a HA LLM Tool to Bedrock toolSpec format."""
    schema: dict[str, Any] = convert(tool.parameters) if tool.parameters else {}
    # Bedrock requires the schema to have an explicit object type and properties
    if "type" not in schema:
        schema["type"] = "object"
    if "properties" not in schema:
        schema["properties"] = {}

    return {
        "toolSpec": {
            "name": bedrock_name,
            "description": tool.description or f"Tool: {tool.name}",
            "inputSchema": {"json": schema},
        }
    }


def _build_tool_config(
    chat_log: ChatLog, supports_tools: bool
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """Return (toolConfig, bedrock_to_ha_name) for the current chat log state."""
    bedrock_to_ha_name: dict[str, str] = {}
    if not (supports_tools and chat_log.llm_api and chat_log.llm_api.tools):
        return None, bedrock_to_ha_name

    tool_specs = []
    for t in chat_log.llm_api.tools:
        safe = _bedrock_tool_name(t.name)
        bedrock_to_ha_name[safe] = t.name
        tool_specs.append(_format_tool(t, safe))

    return {"tools": tool_specs, "toolChoice": {"auto": {}}}, bedrock_to_ha_name


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
                                # Use Bedrock-safe name (hyphens → underscores)
                                # to keep conversation history consistent with
                                # the toolSpec names we registered.
                                "name": _bedrock_tool_name(tc.tool_name),
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
                    "content": [{"text": json.dumps(item.tool_result, cls=_ToolResultEncoder)}],
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

    def _converse_kwargs(
        self,
        chat_log: ChatLog,
        model_id: str,
        max_tokens: int,
        temperature: float,
        supports_tools: bool,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Build the kwargs dict for a Bedrock converse/converse_stream call.

        Returns (kwargs, bedrock_to_ha_name) where bedrock_to_ha_name maps
        Bedrock-safe tool names back to the original HA tool names.
        """
        system, messages = _convert_messages(chat_log.content)
        tool_config, bedrock_to_ha_name = _build_tool_config(chat_log, supports_tools)

        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = system
        if tool_config:
            kwargs["toolConfig"] = tool_config

        return kwargs, bedrock_to_ha_name

    async def _async_stream_from_bedrock(
        self,
        converse_kwargs: dict[str, Any],
        bedrock_to_ha_name: dict[str, str],
    ) -> AsyncIterator[AssistantContentDeltaDict]:
        """Bridge Bedrock's sync converse_stream to an async delta iterator.

        Converts Bedrock stream events into AssistantContentDeltaDict items:
        - text deltas are yielded immediately as {"role": "assistant", "content": "..."}
        - tool use blocks are buffered and yielded complete as {"tool_calls": [...]}
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _run_stream() -> None:
            try:
                response = self._client.converse_stream(**converse_kwargs)
                for event in response["stream"]:
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            except ClientError as err:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", err))
            except Exception as err:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", err))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        executor_task = asyncio.ensure_future(
            self.hass.async_add_executor_job(_run_stream)
        )

        # State for the tool use block currently being assembled
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        current_tool_input: str = ""
        # role must only appear in the first delta; subsequent deltas omit it so
        # HA's Assist UI accumulates them into one message bubble instead of
        # creating a new bubble for every token.
        role_sent = False

        try:
            while True:
                kind, value = await queue.get()

                if kind == "done":
                    break

                if kind == "error":
                    if isinstance(value, ClientError):
                        error_code = value.response.get("Error", {}).get("Code", "")
                        error_msg = value.response.get("Error", {}).get(
                            "Message", str(value)
                        )
                        _LOGGER.error(
                            "AWS Bedrock API error [%s]: %s", error_code, error_msg
                        )
                        raise HomeAssistantError(
                            f"AWS Bedrock error ({error_code}): {error_msg}"
                        ) from value
                    raise HomeAssistantError(
                        f"AWS Bedrock stream error: {value}"
                    ) from value

                event: dict[str, Any] = value

                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        current_tool_id = start["toolUse"]["toolUseId"]
                        current_tool_name = start["toolUse"]["name"]
                        current_tool_input = ""

                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        if not role_sent:
                            yield {"role": "assistant", "content": delta["text"]}
                            role_sent = True
                        else:
                            yield {"content": delta["text"]}
                    elif "toolUse" in delta:
                        current_tool_input += delta["toolUse"].get("input", "")

                elif "contentBlockStop" in event:
                    if current_tool_id and current_tool_name:
                        try:
                            tool_args = (
                                json.loads(current_tool_input)
                                if current_tool_input
                                else {}
                            )
                        except json.JSONDecodeError:
                            tool_args = {}
                        ha_name = bedrock_to_ha_name.get(
                            current_tool_name, current_tool_name
                        )
                        yield {
                            "tool_calls": [
                                llm.ToolInput(
                                    id=current_tool_id,
                                    tool_name=ha_name,
                                    tool_args=tool_args,
                                )
                            ]
                        }
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_input = ""

                elif "metadata" in event:
                    if usage := event["metadata"].get("usage"):
                        _LOGGER.debug(
                            "Token usage — input: %s, output: %s",
                            usage.get("inputTokens"),
                            usage.get("outputTokens"),
                        )
        finally:
            await executor_task

    async def _async_handle_chat_log(self, chat_log: ChatLog) -> None:
        """Generate a streaming response using the AWS Bedrock Converse API.

        Uses converse_stream so that HA's TTS pipeline receives tokens as they
        arrive, dramatically reducing time-to-first-audio for voice assistants.
        Implements an agentic tool-use loop: the model may call Home Assistant
        tools multiple times before producing its final text response.
        """
        options = self.subentry.data
        model_id: str = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens: int = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        temperature: float = options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        supports_tools = _model_supports_tools(model_id)

        for iteration in range(MAX_TOOL_ITERATIONS):
            kwargs, bedrock_to_ha_name = self._converse_kwargs(
                chat_log, model_id, max_tokens, temperature, supports_tools
            )
            stream = self._async_stream_from_bedrock(kwargs, bedrock_to_ha_name)

            has_tool_calls = False
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id, stream
            ):
                if isinstance(content, AssistantContent) and content.tool_calls:
                    has_tool_calls = True

            if not has_tool_calls:
                break

            _LOGGER.debug(
                "Tool-use loop iteration %d/%d (model=%s)",
                iteration + 1,
                MAX_TOOL_ITERATIONS,
                model_id,
            )
