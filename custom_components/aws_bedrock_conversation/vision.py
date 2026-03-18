"""Image analysis service for AWS Bedrock Conversation integration."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from typing import Any

import boto3
from botocore.config import Config

from homeassistant.components.camera import async_get_image
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import ServiceValidationError

from .const import (
    CONF_AWS_ACCESS_KEY_ID,
    CONF_AWS_REGION,
    CONF_AWS_SECRET_ACCESS_KEY,
    CONF_CHAT_MODEL,
    DEFAULT_CHAT_MODEL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Describe what you see in this image concisely and specifically."

# Bedrock-supported image formats and their MIME types
_MIME_TO_BEDROCK_FORMAT: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _bedrock_format_from_path(path: str) -> str:
    """Guess the Bedrock image format string from a file path."""
    mime, _ = mimetypes.guess_type(path)
    return _MIME_TO_BEDROCK_FORMAT.get(mime or "", "jpeg")


async def async_analyze_image(call: ServiceCall) -> dict[str, Any]:
    """Analyze an image using AWS Bedrock and return a text description.

    Accepts either a camera entity ID (grabs the current frame) or a file
    path to an image already saved on disk.  Returns {"description": "..."}.
    """
    hass = call.hass
    camera_entity_id: str | None = call.data.get("camera_entity_id")
    image_path: str | None = call.data.get("image_path")
    prompt: str = call.data.get("prompt") or _DEFAULT_PROMPT

    if not camera_entity_id and not image_path:
        raise ServiceValidationError(
            "Provide either camera_entity_id or image_path.",
            translation_domain=DOMAIN,
            translation_key="analyze_image_no_source",
        )

    # --- get image bytes and format ---
    if camera_entity_id:
        try:
            image = await async_get_image(hass, camera_entity_id)
        except Exception as err:
            raise ServiceValidationError(
                f"Could not capture image from {camera_entity_id}: {err}",
                translation_domain=DOMAIN,
                translation_key="analyze_image_camera_error",
            ) from err
        image_bytes = image.content
        bedrock_format = _MIME_TO_BEDROCK_FORMAT.get(
            image.content_type or "", "jpeg"
        )
    else:
        assert image_path is not None
        if not os.path.isfile(image_path):
            raise ServiceValidationError(
                f"Image file not found: {image_path}",
                translation_domain=DOMAIN,
                translation_key="analyze_image_file_not_found",
            )
        try:
            image_bytes = await hass.async_add_executor_job(
                _read_file, image_path
            )
        except OSError as err:
            raise ServiceValidationError(
                f"Could not read image file: {err}",
                translation_domain=DOMAIN,
                translation_key="analyze_image_file_error",
            ) from err
        bedrock_format = _bedrock_format_from_path(image_path)

    # --- find the first loaded config entry ---
    entry = next(
        (
            e
            for e in hass.config_entries.async_entries(DOMAIN)
            if e.runtime_data is not None
        ),
        None,
    )
    if entry is None:
        raise ServiceValidationError(
            "No loaded AWS Bedrock config entry found.",
            translation_domain=DOMAIN,
            translation_key="analyze_image_no_entry",
        )

    model_id = DEFAULT_CHAT_MODEL
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "conversation":
            model_id = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            break

    # --- call Bedrock ---
    def _run_converse() -> str:
        client: boto3.client = boto3.client(
            "bedrock-runtime",
            region_name=entry.data[CONF_AWS_REGION],
            aws_access_key_id=entry.data[CONF_AWS_ACCESS_KEY_ID],
            aws_secret_access_key=entry.data[CONF_AWS_SECRET_ACCESS_KEY],
            config=Config(connect_timeout=10, read_timeout=60),
        )
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": bedrock_format,
                                "source": {"bytes": image_bytes},
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 512, "temperature": 0.3},
        )
        text_parts = [
            block["text"]
            for block in response["output"]["message"].get("content", [])
            if "text" in block
        ]
        return "".join(text_parts)

    description = await hass.async_add_executor_job(_run_converse)

    _LOGGER.debug("Image analysis result (model=%s): %s", model_id, description)
    return {"description": description}


def _read_file(path: str) -> bytes:
    """Read a file from disk. Runs in an executor."""
    with open(path, "rb") as fh:
        return fh.read()
