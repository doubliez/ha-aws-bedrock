"""Constants for AWS Bedrock Conversation integration."""

import logging

DOMAIN = "aws_bedrock_conversation"
LOGGER = logging.getLogger(__package__)

# Configuration keys
CONF_AWS_ACCESS_KEY_ID = "aws_access_key_id"
CONF_AWS_SECRET_ACCESS_KEY = "aws_secret_access_key"
CONF_AWS_REGION = "aws_region"
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_PROMPT = "prompt"
CONF_RECOMMENDED = "recommended"

# Defaults
DEFAULT_CONVERSATION_NAME = "AWS Bedrock Conversation"
DEFAULT_AI_TASK_NAME = "AWS Bedrock AI Task"
DEFAULT_CHAT_MODEL = "amazon.nova-pro-v1:0"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7

# Max agentic loop iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 10

# AWS Regions where Bedrock is available
AWS_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "sa-east-1",
    "me-south-1",
    "af-south-1",
]

# Available Bedrock model IDs grouped by provider.
# These are text/chat models suitable for conversation agents.
# Any custom model ID (e.g. cross-region inference profiles like
# "us.amazon.nova-pro-v1:0") can be typed manually in the UI.
BEDROCK_MODELS = [
    # Amazon Nova (latest generation)
    ("amazon.nova-premier-v1:0", "Amazon Nova Premier"),
    ("amazon.nova-pro-v1:0", "Amazon Nova Pro"),
    ("amazon.nova-lite-v1:0", "Amazon Nova Lite"),
    ("amazon.nova-micro-v1:0", "Amazon Nova Micro"),
    # Anthropic Claude (newest first)
    ("anthropic.claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("anthropic.claude-opus-4-6-v1", "Claude Opus 4.6"),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5"),
    ("anthropic.claude-opus-4-5-20251101-v1:0", "Claude Opus 4.5"),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5"),
    ("anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4"),
    ("anthropic.claude-3-5-haiku-20241022-v1:0", "Claude 3.5 Haiku"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
    # Meta Llama 4
    ("meta.llama4-maverick-17b-instruct-v1:0", "Llama 4 Maverick 17B"),
    ("meta.llama4-scout-17b-instruct-v1:0", "Llama 4 Scout 17B"),
    # Meta Llama 3.x
    ("meta.llama3-3-70b-instruct-v1:0", "Llama 3.3 70B Instruct"),
    ("meta.llama3-2-90b-instruct-v1:0", "Llama 3.2 90B Instruct"),
    ("meta.llama3-2-11b-instruct-v1:0", "Llama 3.2 11B Instruct"),
    ("meta.llama3-1-405b-instruct-v1:0", "Llama 3.1 405B Instruct"),
    ("meta.llama3-1-70b-instruct-v1:0", "Llama 3.1 70B Instruct"),
    ("meta.llama3-1-8b-instruct-v1:0", "Llama 3.1 8B Instruct"),
    # Mistral
    ("mistral.mistral-large-3-675b-instruct", "Mistral Large 3 (675B)"),
    ("mistral.pixtral-large-2502-v1:0", "Pixtral Large"),
    ("mistral.mistral-large-2407-v1:0", "Mistral Large 2407"),
    # DeepSeek
    ("deepseek.r1-v1:0", "DeepSeek R1"),
    ("deepseek.v3-v1:0", "DeepSeek V3"),
    # Qwen
    ("qwen.qwen3-235b-a22b-2507-v1:0", "Qwen3 235B"),
    ("qwen.qwen3-32b-v1:0", "Qwen3 32B"),
    # Cohere
    ("cohere.command-r-plus-v1:0", "Cohere Command R+"),
    ("cohere.command-r-v1:0", "Cohere Command R"),
    # AI21
    ("ai21.jamba-1-5-large-v1:0", "AI21 Jamba 1.5 Large"),
    ("ai21.jamba-1-5-mini-v1:0", "AI21 Jamba 1.5 Mini"),
    # Writer
    ("writer.palmyra-x5-v1:0", "Writer Palmyra X5"),
]

# Model ID prefixes that support tool use (function calling) via the Converse API.
# Models NOT in this list will have tools disabled to avoid API errors.
TOOL_USE_SUPPORTED_MODELS = [
    "amazon.nova",
    "anthropic.claude",
    "meta.llama4",
    "meta.llama3-3",
    "meta.llama3-2",
    "meta.llama3-1",
    "mistral.mistral-large",
    "mistral.pixtral",
    "cohere.command-r",
    "qwen.qwen3",
    "writer.palmyra",
]
