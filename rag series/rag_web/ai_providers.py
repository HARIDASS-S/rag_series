
import os

from openai import OpenAI
from google import genai
from anthropic import Anthropic
from ollama import Client as OllamaClient

from config import OLLAMA_HOST


PROVIDERS = [
    "Ollama",
    "OpenAI",
    "Google Gemini",
    "Anthropic Claude",
    "DeepSeek",
    "Mistral",
    "Groq",
    "OpenRouter",
    "Together AI",
]


# API-compatible providers.
# Model availability depends on the account and provider.
OPENAI_COMPATIBLE = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY",
    },
}


def get_client(provider, api_key=""):
    """Create a client for the selected provider."""

    if provider == "Ollama":
        return OllamaClient(host=OLLAMA_HOST)

    if provider == "OpenAI":
        return OpenAI(api_key=api_key)

    if provider == "Google Gemini":
        return genai.Client(api_key=api_key)

    if provider == "Anthropic Claude":
        return Anthropic(api_key=api_key)

    if provider in OPENAI_COMPATIBLE:
        if not api_key:
            raise ValueError("Enter the API key for this provider.")

        return OpenAI(
            api_key=api_key,
            base_url=OPENAI_COMPATIBLE[provider]["base_url"],
        )

    raise ValueError(f"Unsupported provider: {provider}")


def list_models(provider, api_key=""):
    """Fetch models from the selected provider."""

    client = get_client(provider, api_key)

    if provider == "Ollama":
        response = client.list()
        models = []

        for item in response.get("models", []):
            name = getattr(item, "model", None)

            if not name and isinstance(item, dict):
                name = item.get("model") or item.get("name")

            if name:
                models.append(name)

        return sorted(set(models))

    if provider == "Google Gemini":
        names = []

        for item in client.models.list():
            name = getattr(item, "name", None)

            if name:
                name = name.removeprefix("models/")
                names.append(name)

        return sorted(set(names))

    if provider == "Anthropic Claude":
        response = client.models.list(limit=100)

        return sorted({
            item.id for item in response.data
        })

    # OpenAI and compatible API providers.
    response = client.models.list()

    return sorted({
        item.id for item in response.data
    })


def generate_answer(
    provider,
    model,
    api_key,
    system_prompt,
    user_prompt,
    history=None,
):
    """Generate an answer with the chosen AI model."""

    history = history or []
    client = get_client(provider, api_key)

    messages = []

    for item in history[-8:]:
        role = item.get("role")
        content = item.get("content")

        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({
                "role": role,
                "content": content
            })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    if provider == "Ollama":
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
        )

        return response["message"]["content"]

    if provider == "Google Gemini":
        # Gemini receives conversation history as text.
        conversation = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages
        )

        response = client.models.generate_content(
            model=model,
            contents=(
                f"{system_prompt}\n\n"
                f"Conversation and question:\n{conversation}"
            ),
        )

        return response.text or "The model returned no text."

    if provider == "Anthropic Claude":
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )

        return "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

    # OpenAI and compatible providers.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
    )

    return response.choices[0].message.content or ""