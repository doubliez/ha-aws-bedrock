"""Custom LLM tools exposed to Home Assistant's conversation agents."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import voluptuous as vol
import yaml

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

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
                "- Use get_automations to read automation configs and help debug, "
                "explain, or improve them. Accepts an optional search filter.\n"
                "Only call these tools when the user explicitly asks you to examine "
                "or improve their Home Assistant setup."
            ),
            llm_context=llm_context,
            tools=[GetDashboardsTool(), GetAutomationsTool()],
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
    ) -> dict[str, Any]:
        """Load and return all Lovelace dashboard configurations."""
        dashboards = await hass.async_add_executor_job(
            _load_lovelace_configs, hass.config.config_dir
        )
        return {"dashboards": dashboards}


class GetAutomationsTool(llm.Tool):
    """Returns Home Assistant automation configurations for analysis."""

    name = "get_automations"
    description = (
        "Returns Home Assistant automation configurations. "
        "Use this when the user asks you to review, debug, explain, or improve "
        "their automations. Pass a search string to filter by name or description "
        "when the user mentions a specific automation or topic."
    )
    parameters = vol.Schema(
        {
            vol.Optional(
                "search",
                description=(
                    "Case-insensitive substring to filter automations by alias or "
                    "description. Omit to return all automations."
                ),
            ): str,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Load automations and enrich with live entity state."""
        search: str = (tool_input.tool_args.get("search") or "").lower().strip()

        automations = await hass.async_add_executor_job(
            _load_automation_configs, hass.config.config_dir
        )

        # Build id → state map from live automation entities so we can add
        # last_triggered and enabled status without a separate tool call.
        state_by_id: dict[str, Any] = {}
        for state in hass.states.async_all("automation"):
            auto_id = state.attributes.get("id")
            if auto_id:
                state_by_id[auto_id] = state

        enriched = []
        for auto in automations:
            auto_id = auto.get("id")
            if auto_id and auto_id in state_by_id:
                s = state_by_id[auto_id]
                auto["enabled"] = s.state == "on"
                auto["last_triggered"] = s.attributes.get("last_triggered")
            enriched.append(auto)

        if search:
            enriched = [
                a
                for a in enriched
                if search in (a.get("alias") or "").lower()
                or search in (a.get("description") or "").lower()
            ]

        return {"count": len(enriched), "automations": enriched}


def _load_automation_configs(config_dir: str) -> list[dict[str, Any]]:
    """Load automation configs from storage and/or YAML.

    Checks .storage/core.automation (UI-created automations) first, then
    falls back to automations.yaml for YAML-mode setups.
    Runs in an executor since it does file I/O.
    """
    automations: list[dict[str, Any]] = []

    # --- storage-mode automations ---
    storage_path = os.path.join(config_dir, ".storage", "core.automation")
    if os.path.isfile(storage_path):
        try:
            with open(storage_path) as fh:
                raw = json.load(fh)
            items = raw.get("data", {}).get("items", [])
            automations.extend(items)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Could not read core.automation storage: %s", err)

    # --- YAML-mode automations (automations.yaml) ---
    yaml_path = os.path.join(config_dir, "automations.yaml")
    if os.path.isfile(yaml_path) and not automations:
        try:
            with open(yaml_path) as fh:
                items = yaml.safe_load(fh) or []
            if isinstance(items, list):
                automations.extend(items)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Could not read automations.yaml: %s", err)

    return automations


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
