# AWS Bedrock Conversation for Home Assistant

A [Home Assistant](https://www.home-assistant.io/) custom integration that connects [Amazon Bedrock](https://aws.amazon.com/bedrock/) foundation models to Home Assistant's built-in conversation and voice assistant pipeline.

Use Claude, Llama, Mistral, Amazon Nova, and many more models from AWS Bedrock to power your smart home assistant — with full support for controlling entities, querying sensors, and all other Home Assistant AI features.

---

## Features

- **Multi-model support** — Claude 3.5, Amazon Nova, Llama 3, Mistral, Cohere, and more
- **Home Assistant control** — Lets the AI turn on lights, read sensors, call services, and interact with all HA entities via the built-in LLM API
- **Agentic tool-use loop** — Supports multi-step reasoning where the model can call HA tools multiple times before giving a final response
- **Configurable per-agent** — Set a custom system prompt, model, temperature, and token limits for each conversation agent
- **Multiple agents** — Add multiple conversation agents with different models/settings
- **HACS compatible**

---

## Requirements

- Home Assistant 2024.6.0 or newer
- An AWS account with Bedrock access
- An IAM user or role with the following permissions:
  - `bedrock:InvokeModel`
  - `bedrock:ListFoundationModels`
- The models you want to use must be **enabled** in your AWS Bedrock console (go to *Model access* → *Manage model access*)

---

## Installation

### Via HACS (recommended)

1. Open HACS in your Home Assistant instance
2. Go to **Integrations** → click the three-dot menu → **Custom repositories**
3. Add `https://github.com/doubliez/ha-aws-bedrock` as an **Integration**
4. Search for "AWS Bedrock Conversation" and install it
5. Restart Home Assistant

### Manual

1. Download the latest release from [GitHub](https://github.com/doubliez/ha-aws-bedrock/releases)
2. Copy the `custom_components/aws_bedrock_conversation` folder to your HA `config/custom_components/` directory
3. Restart Home Assistant

---

## Setup

### 1. Create an AWS IAM user

1. Open the [IAM console](https://console.aws.amazon.com/iam/)
2. Create a new user (or use an existing one)
3. Attach the following inline or managed policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    }
  ]
}
```

4. Create an **Access Key** for programmatic access and save the Access Key ID and Secret Access Key.

### 2. Enable models in AWS Bedrock

1. Open the [Bedrock console](https://console.aws.amazon.com/bedrock/) in your target region
2. Go to **Model access** → **Manage model access**
3. Enable the models you want to use (e.g. *Amazon Nova Lite*, *Claude 3.5 Haiku*)
4. Wait for access to be granted (usually instant for Amazon/Meta models, may take minutes for Anthropic)

### 3. Add the integration in Home Assistant

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **AWS Bedrock Conversation**
3. Enter your:
   - AWS Access Key ID
   - AWS Secret Access Key
   - AWS Region (choose the region where you enabled models)
4. Click **Submit**

A default **AWS Bedrock Conversation** agent will be created automatically.

### 4. Use with the Voice Assistant

1. Go to **Settings** → **Voice Assistants**
2. Select (or create) a voice assistant pipeline
3. Set the **Conversation agent** to **AWS Bedrock Conversation**

---

## Configuration

### Conversation Agent Options

After setup, click **Configure** on the integration card to access the conversation subentry settings:

| Option | Description | Default |
|--------|-------------|---------|
| System prompt | Custom instructions for the AI | HA default |
| Home Assistant APIs | Enable entity control tools | Assist API |
| Use recommended settings | Use default model and parameters | `true` |

With **recommended settings** disabled, you can also configure:

| Option | Description | Default |
|--------|-------------|---------|
| Model | Bedrock model ID to use | `amazon.nova-lite-v1:0` |
| Max output tokens | Maximum response length | `2048` |
| Temperature | Response randomness (0–1) | `0.7` |
| Top P | Nucleus sampling (0–1) | `0.9` |

### Available Models

The following models are pre-listed (any model ID can be typed manually):

| Provider | Model | ID |
|----------|-------|----|
| Amazon | Nova Premier | `amazon.nova-premier-v1:0` |
| Amazon | Nova Pro *(default)* | `amazon.nova-pro-v1:0` |
| Amazon | Nova Lite | `amazon.nova-lite-v1:0` |
| Amazon | Nova Micro | `amazon.nova-micro-v1:0` |
| Anthropic | Claude Sonnet 4.6 | `anthropic.claude-sonnet-4-6` |
| Anthropic | Claude Opus 4.6 | `anthropic.claude-opus-4-6-v1` |
| Anthropic | Claude Sonnet 4.5 | `anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Anthropic | Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| Anthropic | Claude 3.5 Haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` |
| Meta | Llama 4 Maverick 17B | `meta.llama4-maverick-17b-instruct-v1:0` |
| Meta | Llama 4 Scout 17B | `meta.llama4-scout-17b-instruct-v1:0` |
| Meta | Llama 3.3 70B | `meta.llama3-3-70b-instruct-v1:0` |
| Meta | Llama 3.1 405B | `meta.llama3-1-405b-instruct-v1:0` |
| Mistral | Mistral Large 3 | `mistral.mistral-large-3-675b-instruct` |
| Mistral | Pixtral Large | `mistral.pixtral-large-2502-v1:0` |
| DeepSeek | DeepSeek R1 | `deepseek.r1-v1:0` |
| DeepSeek | DeepSeek V3 | `deepseek.v3-v1:0` |
| Qwen | Qwen3 235B | `qwen.qwen3-235b-a22b-2507-v1:0` |
| Cohere | Command R+ | `cohere.command-r-plus-v1:0` |
| AI21 | Jamba 1.5 Large | `ai21.jamba-1-5-large-v1:0` |
| Writer | Palmyra X5 | `writer.palmyra-x5-v1:0` |
| ... and more | See `const.py` | |

> **Note:** Cross-region inference profiles (e.g. `us.amazon.nova-pro-v1:0`) are also supported — just type the full ID.

---

## Troubleshooting

### "Access denied" during setup

Your IAM user/role is missing the `bedrock:ListFoundationModels` permission. Add it and try again.

### "Model not found" or `ValidationException`

The model you selected is not enabled in your AWS account or region. Go to the [Bedrock console](https://console.aws.amazon.com/bedrock/) → **Model access** and enable the model.

### The AI can't control my devices

Make sure **Home Assistant APIs** is set to **Assist** (or another API) in the conversation agent configuration. The model must also support tool use — Amazon Nova, Claude 3.x, Llama 3.x, and Mistral Large all do.

### High latency

- Use a region close to your Home Assistant server
- Use a faster/smaller model (Nova Micro, Claude 3 Haiku, Llama 3.2 1B)
- Reduce **Max output tokens**

---

## Contributing

Pull requests are welcome! Please open an issue first for significant changes.

---

## License

MIT License. See [LICENSE](LICENSE).
