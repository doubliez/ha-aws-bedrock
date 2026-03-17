"""Config flow for AWS Bedrock Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError, EndpointResolutionError, NoCredentialsError
import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_REAUTH,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    AWS_REGIONS,
    BEDROCK_MODELS,
    CONF_AWS_ACCESS_KEY_ID,
    CONF_AWS_REGION,
    CONF_AWS_SECRET_ACCESS_KEY,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)

if TYPE_CHECKING:
    from . import BedrockConfigEntry

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_AWS_ACCESS_KEY_ID): str,
        vol.Required(CONF_AWS_SECRET_ACCESS_KEY): str,
        vol.Required(CONF_AWS_REGION, default="us-east-1"): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(label=r, value=r) for r in AWS_REGIONS
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)

DEFAULT_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

DEFAULT_AI_TASK_OPTIONS = {
    CONF_RECOMMENDED: True,
}


async def _validate_credentials(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate AWS credentials by making a lightweight Bedrock API call."""

    def _do_validate() -> None:
        bedrock_client = boto3.client(
            "bedrock",
            region_name=data[CONF_AWS_REGION],
            aws_access_key_id=data[CONF_AWS_ACCESS_KEY_ID],
            aws_secret_access_key=data[CONF_AWS_SECRET_ACCESS_KEY],
        )
        # ListFoundationModels is a lightweight call that validates credentials and region.
        bedrock_client.list_foundation_models(byOutputModality="TEXT")

    await hass.async_add_executor_job(_do_validate)


class AWSBedrockConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AWS Bedrock Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await _validate_credentials(self.hass, user_input)
            except ClientError as err:
                error_code = err.response.get("Error", {}).get("Code", "")
                if error_code in ("InvalidClientTokenId", "AuthFailure"):
                    errors["base"] = "invalid_credentials"
                elif error_code == "AccessDeniedException":
                    errors["base"] = "access_denied"
                else:
                    _LOGGER.exception("Unexpected AWS ClientError: %s", err)
                    errors["base"] = "cannot_connect"
            except EndpointResolutionError:
                errors["base"] = "invalid_region"
            except NoCredentialsError:
                errors["base"] = "invalid_credentials"
            except Exception:
                _LOGGER.exception("Unexpected exception during AWS Bedrock setup")
                errors["base"] = "unknown"
            else:
                if self.source == SOURCE_REAUTH:
                    return self.async_update_reload_and_abort(
                        self._get_reauth_entry(), data_updates=user_input
                    )
                return self.async_create_entry(
                    title=f"AWS Bedrock ({user_input[CONF_AWS_REGION]})",
                    data=user_input,
                    subentries=[
                        {
                            "subentry_type": "conversation",
                            "data": DEFAULT_CONVERSATION_OPTIONS,
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        },
                    ],
                )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors or None,
            description_placeholders={
                "docs_url": "https://github.com/doubliez/ha-aws-bedrock#setup",
            },
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Perform reauth upon an API authentication error."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Dialog that informs the user that reauth is required."""
        if not user_input:
            return self.async_show_form(
                step_id="reauth_confirm", data_schema=STEP_USER_DATA_SCHEMA
            )
        return await self.async_step_user(user_input)

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: BedrockConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": ConversationSubentryFlowHandler,
        }


class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a new subentry."""
        self.options = DEFAULT_CONVERSATION_OPTIONS.copy()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure basic options."""
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        if suggested_llm_apis := self.options.get(CONF_LLM_HASS_API):
            if isinstance(suggested_llm_apis, str):
                suggested_llm_apis = [suggested_llm_apis]
            known_apis = {api.id for api in llm.async_get_apis(self.hass)}
            self.options[CONF_LLM_HASS_API] = [
                api for api in suggested_llm_apis if api in known_apis
            ]

        step_schema: VolDictType = {}
        errors: dict[str, str] = {}

        if self._is_new:
            step_schema[vol.Required(CONF_NAME, default=DEFAULT_CONVERSATION_NAME)] = str

        step_schema.update(
            {
                vol.Optional(CONF_PROMPT): TemplateSelector(),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
                vol.Required(
                    CONF_RECOMMENDED,
                    default=self.options.get(CONF_RECOMMENDED, False),
                ): bool,
            }
        )

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)

            if user_input[CONF_RECOMMENDED]:
                if not errors:
                    if self._is_new:
                        return self.async_create_entry(
                            title=user_input.pop(CONF_NAME),
                            data=user_input,
                        )
                    return self.async_update_and_abort(
                        self._get_entry(),
                        self._get_reconfigure_subentry(),
                        data=user_input,
                    )
            else:
                self.options.update(user_input)
                if (
                    CONF_LLM_HASS_API in self.options
                    and CONF_LLM_HASS_API not in user_input
                ):
                    self.options.pop(CONF_LLM_HASS_API)
                if not errors:
                    return await self.async_step_advanced()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
            errors=errors or None,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure advanced model options."""
        errors: dict[str, str] = {}

        model_options = [
            SelectOptionDict(label=label, value=model_id)
            for model_id, label in BEDROCK_MODELS
        ]

        step_schema: VolDictType = {
            vol.Required(
                CONF_CHAT_MODEL,
                default=self.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    custom_value=True,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=self.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): int,
            vol.Optional(
                CONF_TEMPERATURE,
                default=self.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        }

        if user_input is not None:
            self.options.update(user_input)

            if not errors:
                if self._is_new:
                    return self.async_create_entry(
                        title=self.options.pop(CONF_NAME, DEFAULT_CONVERSATION_NAME),
                        data=self.options,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=self.options,
                )

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(step_schema), self.options
            ),
            errors=errors or None,
            last_step=True,
        )
