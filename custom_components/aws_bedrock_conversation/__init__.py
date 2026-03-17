"""AWS Bedrock Conversation integration for Home Assistant."""

from __future__ import annotations

import boto3

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    CONF_AWS_ACCESS_KEY_ID,
    CONF_AWS_REGION,
    CONF_AWS_SECRET_ACCESS_KEY,
)

PLATFORMS = [Platform.CONVERSATION]

type BedrockConfigEntry = ConfigEntry[boto3.client]


async def async_setup_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Set up AWS Bedrock from a config entry."""

    def _create_client() -> boto3.client:
        return boto3.client(
            "bedrock-runtime",
            region_name=entry.data[CONF_AWS_REGION],
            aws_access_key_id=entry.data[CONF_AWS_ACCESS_KEY_ID],
            aws_secret_access_key=entry.data[CONF_AWS_SECRET_ACCESS_KEY],
        )

    try:
        client = await hass.async_add_executor_job(_create_client)
    except Exception as err:
        raise ConfigEntryNotReady(f"Failed to create AWS Bedrock client: {err}") from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
