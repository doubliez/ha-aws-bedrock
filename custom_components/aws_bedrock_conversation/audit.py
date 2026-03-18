"""Audit service for AWS Bedrock Conversation integration."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config
from homeassistant.components import persistent_notification
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .const import (
    CONF_AWS_ACCESS_KEY_ID,
    CONF_AWS_REGION,
    CONF_AWS_SECRET_ACCESS_KEY,
    CONF_CHAT_MODEL,
    DEFAULT_CHAT_MODEL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# How often to refresh the in-progress notification with streamed text (seconds)
_PROGRESS_UPDATE_INTERVAL = 4

# Allow up to 5 minutes for the stream to complete on large instances
_AUDIT_READ_TIMEOUT = 300

_AUDIT_PROMPT = """\
You are a Home Assistant expert performing a setup audit.
Analyze the entity and device data below and provide specific, actionable recommendations.

Today's date: {today}

## Entities ({entity_count} total)

{entity_data}

## Devices ({device_count} total)

{device_data}

---

Provide a structured audit report with the following sections.
Be specific — use exact entity_id values. Focus on the most impactful issues only.

### Entities to Consider Deleting
Stale (unavailable or unknown for many days), orphaned (no linked device), or clearly \
redundant entities. Format each as: `entity_id` — reason

### Rename Suggestions
Entities with auto-generated, unclear, or confusing names (e.g. "light.light_3", \
"switch.switch_1_2"). Format each as: `entity_id` — suggested name — reason

### Potential Issues
Devices or entities with configuration problems worth investigating.

### General Observations
Brief overall assessment of the setup quality and organisation.
"""


def _days_ago(dt: datetime) -> int:
    """Return how many days ago a datetime was."""
    return (datetime.now(UTC) - dt.astimezone(UTC)).days


def _collect_entity_data(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Build a compact, audit-focused summary of all entities."""
    entity_reg = er.async_get(hass)
    device_reg = dr.async_get(hass)
    rows = []

    for state in hass.states.async_all():
        reg_entry = entity_reg.async_get(state.entity_id)

        device_name: str | None = None
        if reg_entry and reg_entry.device_id:
            device = device_reg.async_get(reg_entry.device_id)
            if device:
                device_name = device.name_by_user or device.name

        rows.append(
            {
                "entity_id": state.entity_id,
                "friendly_name": state.attributes.get("friendly_name") or state.entity_id,
                "state": state.state,
                "last_changed_days_ago": _days_ago(state.last_changed),
                "platform": reg_entry.platform if reg_entry else None,
                "disabled": bool(reg_entry and reg_entry.disabled_by),
                "hidden": bool(reg_entry and reg_entry.hidden_by),
                "device": device_name,
            }
        )

    return rows


def _collect_device_data(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Build a compact, audit-focused summary of all devices."""
    device_reg = dr.async_get(hass)
    entity_reg = er.async_get(hass)
    rows = []

    for device in device_reg.devices.values():
        entity_count = len(er.async_entries_for_device(entity_reg, device.id))
        rows.append(
            {
                "name": device.name_by_user or device.name or device.id,
                "manufacturer": device.manufacturer,
                "model": device.model,
                "entity_count": entity_count,
                "disabled": bool(device.disabled_by),
            }
        )

    return rows


async def _stream_response(
    hass: HomeAssistant,
    client: Any,
    model_id: str,
    prompt: str,
) -> str:
    """Stream the Bedrock response, updating the notification every few seconds.

    Uses converse_stream so the user sees the report being written live rather
    than staring at a blank "Running…" screen for minutes.
    """
    accumulated: list[str] = []
    errors: list[BaseException] = []

    def _run_stream() -> None:
        try:
            response = client.converse_stream(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 4096, "temperature": 0.3},
            )
            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        accumulated.append(delta["text"])
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    stream_task = hass.async_add_executor_job(_run_stream)

    # Poll every few seconds: if the task is still running, push an in-progress
    # notification so the user can see text appearing as it's generated.
    while True:
        done, _ = await asyncio.wait({asyncio.ensure_future(stream_task)}, timeout=_PROGRESS_UPDATE_INTERVAL)
        if done:
            break
        if accumulated:
            persistent_notification.async_create(
                hass,
                message="*Generating report…*\n\n" + "".join(accumulated),
                title="AWS Bedrock Audit — In Progress",
                notification_id="aws_bedrock_audit",
            )

    await stream_task

    if errors:
        raise errors[0]

    return "".join(accumulated)


async def async_run_audit(call: ServiceCall) -> None:
    """Run a full Home Assistant setup audit using AWS Bedrock."""
    hass = call.hass

    # Find the first loaded config entry
    entry = next(
        (
            e
            for e in hass.config_entries.async_entries(DOMAIN)
            if e.runtime_data is not None
        ),
        None,
    )
    if entry is None:
        _LOGGER.error("No loaded AWS Bedrock config entry found for audit")
        return

    # Dedicated client with an extended read timeout — streaming responses
    # for large payloads can take several minutes.
    def _create_audit_client() -> boto3.client:
        return boto3.client(
            "bedrock-runtime",
            region_name=entry.data[CONF_AWS_REGION],
            aws_access_key_id=entry.data[CONF_AWS_ACCESS_KEY_ID],
            aws_secret_access_key=entry.data[CONF_AWS_SECRET_ACCESS_KEY],
            config=Config(connect_timeout=10, read_timeout=_AUDIT_READ_TIMEOUT),
        )

    client = await hass.async_add_executor_job(_create_audit_client)

    # Pick the model from the first conversation subentry, fall back to default
    model_id = DEFAULT_CHAT_MODEL
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "conversation":
            model_id = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            break

    entity_rows = _collect_entity_data(hass)
    device_rows = _collect_device_data(hass)
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    prompt = _AUDIT_PROMPT.format(
        today=today,
        entity_count=len(entity_rows),
        entity_data=json.dumps(entity_rows, separators=(",", ":")),
        device_count=len(device_rows),
        device_data=json.dumps(device_rows, separators=(",", ":")),
    )

    _LOGGER.debug(
        "Starting Bedrock audit: %d entities, %d devices, model=%s",
        len(entity_rows),
        len(device_rows),
        model_id,
    )

    persistent_notification.async_create(
        hass,
        message=(
            f"Analysing **{len(entity_rows)} entities** and "
            f"**{len(device_rows)} devices**. The report will appear here as it generates."
        ),
        title="AWS Bedrock Audit — Running…",
        notification_id="aws_bedrock_audit",
    )

    try:
        report = await _stream_response(hass, client, model_id, prompt)
    except Exception as err:
        _LOGGER.error("Bedrock audit failed: %s", err)
        persistent_notification.async_create(
            hass,
            message=f"The audit failed with an error:\n\n`{err}`",
            title="AWS Bedrock Audit — Failed",
            notification_id="aws_bedrock_audit",
        )
        return

    persistent_notification.async_create(
        hass,
        message=report,
        title="AWS Bedrock Setup Audit",
        notification_id="aws_bedrock_audit",
    )

    _LOGGER.info("Bedrock audit complete (%d entities analysed)", len(entity_rows))
