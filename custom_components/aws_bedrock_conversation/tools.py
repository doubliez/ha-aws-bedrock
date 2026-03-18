"""Custom LLM tools exposed to Home Assistant's conversation agents."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.typing import JsonObjectType

_LOGGER = logging.getLogger(__name__)

BEDROCK_TOOLS_API_ID = "aws_bedrock_tools"


class BedrockToolsAPI(llm.API):
    """Custom LLM API that provides Home Assistant analysis tools.

    Appears in the 'Home Assistant APIs' selector of the conversation agent
    configuration alongside the built-in Assist API.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialise the API."""
        super().__init__(
            hass=hass,
            id=BEDROCK_TOOLS_API_ID,
            name="AWS Bedrock Tools",
        )

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return the API instance with all available tools."""
        return llm.APIInstance(
            api=self,
            api_prompt=(
                "You have access to additional Home Assistant analysis tools.\n"
                "- Use get_dashboards to inspect Lovelace dashboard layouts and "
                "suggest improvements, spot broken entity references, or identify "
                "cluttered views.\n"
                "Only call these tools when the user explicitly asks you to examine "
                "or improve their Home Assistant setup."
            ),
            llm_context=llm_context,
            tools=[GetDashboardsTool()],
        )


class GetDashboardsTool(llm.Tool):
    """Returns all Lovelace dashboard configurations for analysis."""

    name = "get_dashboards"
    description = (
        "Returns the full configuration of all Home Assistant Lovelace dashboards. "
        "Use this when the user asks you to review their dashboards, find broken "
        "entity references, suggest a better layout, or identify anything that looks "
        "wrong with their Home Assistant UI setup."
    )
    parameters = vol.Schema({})

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Load and return all Lovelace dashboard configurations."""
        dashboards = await hass.async_add_executor_job(
            _load_lovelace_configs, hass.config.config_dir
        )
        return {"dashboards": dashboards}


def _load_lovelace_configs(config_dir: str) -> dict[str, Any]:
    """Read all Lovelace dashboard JSON files from HA storage.

    HA stores Lovelace configs in .storage/lovelace (default dashboard)
    and .storage/lovelace.<url_path> (additional dashboards).
    Runs in an executor since it does file I/O.
    """
    storage_dir = os.path.join(config_dir, ".storage")

    if not os.path.isdir(storage_dir):
        return {"error": "HA storage directory not found"}

    dashboards: dict[str, Any] = {}

    for filename in sorted(os.listdir(storage_dir)):
        if not filename.startswith("lovelace") or filename.endswith(".bak"):
            continue

        key = "default" if filename == "lovelace" else filename[len("lovelace."):]
        filepath = os.path.join(storage_dir, filename)

        try:
            with open(filepath) as fh:
                raw = json.load(fh)
            dashboards[key] = raw.get("data", {})
        except Exception as err:
            _LOGGER.warning("Could not read dashboard file %s: %s", filename, err)
            dashboards[key] = {"error": str(err)}

    if not dashboards:
        return {
            "message": (
                "No storage-mode Lovelace dashboards found. "
                "You may be using YAML-mode dashboards — those are not "
                "accessible via this tool."
            )
        }

    return dashboards
