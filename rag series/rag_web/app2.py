import shutil
from pathlib import Path

import chromadb
import gradio as gr

from ai_providers import PROVIDERS, list_models
from config import DB_DIR, OLLAMA_HOST, UPLOAD_DIR, COLLECTION_NAME
from ingestion import ingest_files
from rag_engine import ask_question


# ==================================================
# Database setup
# ==================================================

client = chromadb.PersistentClient(path=str(DB_DIR))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Provider and model management
# ==================================================

def refresh_model_choices(provider, api_key):
    """Fetch available models for the selected provider."""

    if not provider:
        return (
            gr.update(choices=[], value=None),
            "Select an AI provider first.",
        )

    if provider != "Ollama" and not (api_key or "").strip():
        return (
            gr.update(choices=[], value=None),
            "Enter your API key, then refresh models.",
        )

    try:
        models = list_models(
            provider=provider,
            api_key=(api_key or "").strip(),
        )

        if not models:
            if provider == "Ollama":
                message = (
                    f"No local models found at `{OLLAMA_HOST}`. "
                    "Run `ollama pull llama3` in PowerShell, "
                    "then refresh models."
                )
            else:
                message = f"No models were returned by {provider}."

            return gr.update(choices=[], value=None), message

        # Preferred model, if available.
        preferred_models = {
            "Ollama": "llama3:latest",
        }

        preferred = preferred_models.get(provider)
        default_model = preferred if preferred in models else models[0]

        return (
            gr.update(choices=models, value=default_model),
            f"Loaded {len(models)} model(s) from {provider}.",
        )

    except Exception as exc:
        if provider == "Ollama":
            message = (
                f"Cannot connect to Ollama at `{OLLAMA_HOST}`.\n\n"
                "1. Start Ollama.\n"
                "2. Check that a model is installed using `ollama list`.\n"
                "3. Click Refresh models again.\n\n"
                f"Details: {exc}"
            )
        else:
            message = f"Unable to fetch models from {provider}: {exc}"

        return gr.update(choices=[], value=None), message


def update_provider(provider):
    """Update API-key visibility and clear stale model choices."""

    is_ollama = provider == "Ollama"

    if is_ollama:
        status = "Ollama runs locally; no API key is required."
    else:
        status = "Enter your API key, then refresh available models."

    return (
        gr.update(
            visible=not is_ollama,
            value="",
            label="API key",
            placeholder="Enter your provider API key",
        ),
        status,
        gr.update(choices=[], value=None),
    )


# ==================================================
# Chat
# ==================================================

def chat(message, history, provider, model, api_key):
    """Answer user questions using the selected AI provider."""

    if not isinstance(message, str) or not message.strip():
        return "Please enter a question."

    if not provider:
        return "Please select an AI provider."

    if not model:
        return "No model selected. Click 'Refresh models' and select a model."

    if provider != "Ollama" and not (api_key or "").strip():
        return "Please enter your API key for the selected provider."

    try:
        answer = ask_question(
            question=message.strip(),
            history=history or [],
            provider=provider,
            model=model,
            api_key=(api_key or "").strip(),
        )

        return str(answer)

    except Exception as exc:
        error_message = str(exc)

        if provider == "Ollama":
            return (
                "AI request failed because Ollama may be unavailable.\n\n"
                f"Expected server: {OLLAMA_HOST}\n\n"
                "Check Ollama with `ollama list`, then try again.\n\n"
                f"Details: {error_message}"
            )

        return f"AI request failed: {error_message}"


# ==================================================
# Document upload and indexing
# ==================================================

def upload_and_index(files, mode):
    """Save uploaded documents and add them to the vector database."""

    if not files:
        return "Please select at least one document.", []

    saved_paths = []

    try:
        for file in files:
            source = Path(
                file.name if hasattr(file, "name") else str(file)
            )

            if not source.is_file():
                raise FileNotFoundError(
                    f"Uploaded file not found: {source.name}"
                )

            destination = UPLOAD_DIR / source.name

            # Avoid copying a file onto itself.
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

            saved_paths.append(str(destination))

        replace_existing = mode == "Replace knowledge base"

        count = ingest_files(
            saved_paths,
            replace=replace_existing,
        )

        filenames = [Path(path).name for path in saved_paths]

        return (
            f"Indexing complete. Database contains {count} chunks.",
            filenames,
        )

    except Exception as exc:
        return f"Indexing failed: {exc}", []


# ==================================================
# Clear knowledge base
# ==================================================

def clear_database():
    """Delete the vector collection and uploaded files."""

    errors = []

    try:
        client.delete_collection(name=COLLECTION_NAME)

    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            errors.append(f"Collection deletion: {exc}")

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for path in UPLOAD_DIR.iterdir():
            if path.is_file():
                path.unlink()

    except Exception as exc:
        errors.append(f"Upload cleanup: {exc}")

    if errors:
        return (
            "Some cleanup operations failed:\n"
            + "\n".join(errors)
        )

    return "Knowledge base cleared. Upload documents to re-index."


# ==================================================
# Application CSS
# ==================================================

APP_CSS = """
:root {
    color-scheme: light;
    --chat-bg: #f9fafb;
    --bubble-user: #e3f2fd;
    --bubble-ai: #ffffff;
    --bubble-border: #d1d5db;
}

body {
    background: var(--chat-bg);
    font-family: "Inter", sans-serif;
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 32px 20px !important;
}

.app-header {
    text-align: center;
    margin-bottom: 24px;
}

.app-header h1 {
    color: #111827;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 6px;
}

.app-header p {
    color: #6b7280;
    font-size: 15px;
    margin-top: 0;
}

.chat-panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    padding: 20px !important;
}

.chat-panel textarea,
.chat-panel input {
    border-radius: 12px !important;
    border: 1px solid #d1d5db;
    padding: 12px;
    font-size: 15px;
}

.gr-button {
    border-radius: 10px !important;
    font-weight: 600;
    padding: 10px 16px;
}

.gr-button-primary {
    background: #2563eb !important;
    color: #ffffff !important;
}

.gr-button-stop {
    background: #dc2626 !important;
    color: #ffffff !important;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 20px 12px !important;
    }

    .chat-panel {
        padding: 16px !important;
    }
}
"""


# ==================================================
# Gradio application
# ==================================================

with gr.Blocks(
    title="RAG Chat Assistant",
    css=APP_CSS,
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # RAG Chat Assistant

        Your AI companion for document-based question answering.
        Upload documents, index them, and ask questions about their content.
        """,
        elem_classes=["app-header"],
    )

    # ----------------------------------------------
    # Provider and model selection
    # ----------------------------------------------

    with gr.Column(elem_classes=["chat-panel"]):

        with gr.Row():
            provider_dropdown = gr.Dropdown(
                label="AI Provider",
                choices=PROVIDERS,
                value=(
                    "Ollama" if "Ollama" in PROVIDERS
                    else (PROVIDERS[0] if PROVIDERS else None)
                ),
                interactive=True,
            )

            api_key_input = gr.Textbox(
                label="API key",
                placeholder="Enter your provider API key",
                type="password",
                value="",
                visible=(
                    not (
                        "Ollama" in PROVIDERS
                    )
                ),
            )

        provider_status = gr.Markdown(
            "Select a provider and refresh available models."
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[],
                value=None,
                interactive=True,
            )

            refresh_button = gr.Button("Refresh models")

        # Load models when the page opens.
        demo.load(
            fn=refresh_model_choices,
            inputs=[provider_dropdown, api_key_input],
            outputs=[model_dropdown, provider_status],
        )

        # Refresh model list on demand.
        refresh_button.click(
            fn=refresh_model_choices,
            inputs=[provider_dropdown, api_key_input],
            outputs=[model_dropdown, provider_status],
        )

        # Update the API-key field when provider changes.
        provider_dropdown.change(
            fn=update_provider,
            inputs=[provider_dropdown],
            outputs=[
                api_key_input,
                provider_status,
                model_dropdown,
            ],
        )

        # Refresh when the user presses Enter in the API-key field.
        api_key_input.submit(
            fn=refresh_model_choices,
            inputs=[provider_dropdown, api_key_input],
            outputs=[model_dropdown, provider_status],
        )

        # ------------------------------------------
        # Chat interface
        # ------------------------------------------

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[
                provider_dropdown,
                model_dropdown,
                api_key_input,
            ],
            chatbot=gr.Chatbot(
                height=500,
                show_label=False,
                placeholder="Ask questions about your documents...",
            ),
            title="Chat",
            description=(
                "Ask questions about your indexed documents "
                "or chat with your selected AI model."
            ),
        )

    # ----------------------------------------------
    # Document upload
    # ----------------------------------------------

    gr.Markdown("## Document Upload")

    files = gr.File(
        label="Upload documents",
        file_count="multiple",
        type="filepath",
        file_types=[
            ".pdf",
            ".txt",
            ".docx",
            ".csv",
            ".xlsx",
        ],
    )

    mode = gr.Radio(
        choices=[
            "Replace knowledge base",
            "Add to knowledge base",
        ],
        value="Replace knowledge base",
        label="Indexing mode",
    )

    with gr.Row():
        index_button = gr.Button(
            "Index documents",
            variant="primary",
        )

        clear_button = gr.Button(
            "Clear knowledge base",
            variant="stop",
        )

    status = gr.Textbox(
        label="Status",
        interactive=False,
        lines=3,
    )

    uploaded_files = gr.JSON(
        label="Files processed",
    )

    index_button.click(
        fn=upload_and_index,
        inputs=[files, mode],
        outputs=[status, uploaded_files],
    )

    clear_button.click(
        fn=clear_database,
        inputs=[],
        outputs=[status],
    )


# ==================================================
# Launch application
# ==================================================

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )