import shutil
from pathlib import Path

import chromadb
import gradio as gr

from ai_providers import PROVIDERS, list_models
from config import DB_DIR, OLLAMA_HOST, UPLOAD_DIR, COLLECTION_NAME
from ingestion import ingest_files
from rag_engine import ask_question


# --------------------------------------------------
# Database setup
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_DIR))

# Ensure the upload directory exists.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Provider and model management
# --------------------------------------------------

def refresh_model_choices(provider, api_key):
    """Fetch models for the selected provider."""

    if not provider:
        return (
            gr.update(choices=[], value=None),
            "Select a provider to load available models.",
        )

    # Ollama generally does not require an API key.
    if provider != "Ollama" and not (api_key or "").strip():
        return (
            gr.update(choices=[], value=None),
            "Enter your API key, then refresh available models.",
        )

    try:
        models = list_models(
            provider=provider,
            api_key=(api_key or "").strip(),
        )

        if not models:
            message = (
                "Connected to Ollama, but no local models are installed. "
                "Install one with `ollama pull llama3`, then refresh."
                if provider == "Ollama"
                else f"No models were returned by {provider}."
            )
            return gr.update(choices=[], value=None), message

        preferred_models = {
            "Ollama": "llama3:latest",
        }

        preferred = preferred_models.get(provider)
        default_model = (
            preferred if preferred in models else models[0]
        )

        return (
            gr.update(
                choices=models,
                value=default_model,
            ),
            f"Loaded {len(models)} models from {provider}.",
        )

    except Exception as exc:
        message = (
            f"Ollama is not reachable at `{OLLAMA_HOST}`. "
            "Start the Ollama app (or run `ollama serve`), then refresh "
            f"models. Details: {exc}"
            if provider == "Ollama"
            else f"Unable to fetch models for {provider}: {exc}"
        )
        return gr.update(choices=[], value=None), message


def update_api_key_visibility(provider):
    """Show a helpful label for local versus cloud providers."""

    is_ollama = provider == "Ollama"

    return (
        gr.update(
            visible=not is_ollama,
            value="",
            label="API key",
            placeholder="Enter your provider API key",
        ),
        gr.update(
            value=(
                "Ollama runs locally; no API key is required."
                if is_ollama
                else "Enter your API key, then refresh available models."
            )
        ),
        gr.update(choices=[], value=None),
    )


# --------------------------------------------------
# Chat
# --------------------------------------------------

def chat(message, history, provider, model, api_key):
    """Answer a question using the selected provider and model."""

    if not isinstance(message, str) or not message.strip():
        return "Please enter a question."

    if not provider:
        return "Please select an AI provider."

    if not model:
        return "No model selected. Click 'Refresh models' and choose a model."

    if provider != "Ollama" and not (api_key or "").strip():
        return "Please enter your API key for the selected provider."

    try:
        return ask_question(
            question=message.strip(),
            history=history or [],
            provider=provider,
            model=model,
            api_key=(api_key or "").strip(),
        )

    except Exception as exc:
        return f"AI request failed: {exc}"


# --------------------------------------------------
# Document upload and indexing
# --------------------------------------------------

def upload_and_index(files, mode):
    """Save uploaded files and add them to the RAG index."""

    if not files:
        return "Please select at least one document.", []

    saved_paths = []

    try:
        for file in files:
            # Gradio type="filepath" normally supplies file paths.
            source = Path(
                file.name if hasattr(file, "name") else str(file)
            )

            if not source.is_file():
                raise FileNotFoundError(
                    f"Uploaded file not found: {source.name}"
                )

            # Avoid directory traversal and keep only the filename.
            destination = UPLOAD_DIR / source.name

            # Avoid overwriting an existing file with different content.
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

            saved_paths.append(str(destination))

        count = ingest_files(
            saved_paths,
            replace=(mode == "Replace knowledge base"),
        )

        filenames = [Path(path).name for path in saved_paths]

        return (
            f"Indexing complete. Database contains {count} chunks.",
            filenames,
        )

    except Exception as exc:
        return f"Indexing failed: {exc}", []


# --------------------------------------------------
# Clear knowledge base
# --------------------------------------------------

def clear_database():
    """Delete the indexed collection and saved uploads."""

    errors = []

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as exc:
        # A missing collection is not necessarily an error.
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
        return "Some cleanup operations failed:\n" + "\n".join(errors)

    return (
        "Knowledge base collection and saved uploads cleared. "
        "The collection may be recreated automatically when indexing resumes."
    )


# --------------------------------------------------
# Gradio application
# --------------------------------------------------

APP_CSS = """
:root {
    color-scheme: light;
}

body {
    background: #f4f6fa;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 28px 32px 40px !important;
}

.app-header {
    margin-bottom: 24px;
}

.app-header h1 {
    color: #17233b;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.04em;
    margin-bottom: 6px;
}

.app-header p {
    color: #667085;
    font-size: 15px;
    margin-top: 0;
}

.workspace-layout {
    align-items: flex-start;
    gap: 22px;
}

.sidebar-panel,
.chat-panel {
    background: #ffffff;
    border: 1px solid #e7eaf0;
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(20, 34, 60, 0.045);
}

.sidebar-panel {
    padding: 22px !important;
}

.chat-panel {
    padding: 22px !important;
}

.sidebar-panel h2,
.sidebar-panel h3,
.chat-panel h2,
.chat-panel h3 {
    color: #17233b;
    letter-spacing: -0.02em;
}

.sidebar-panel .wrap,
.chat-panel .wrap {
    border-radius: 12px;
}

.chat-panel textarea,
.chat-panel input,
.sidebar-panel textarea,
.sidebar-panel input {
    border-radius: 10px !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 18px 14px 28px !important;
    }

    .workspace-layout {
        gap: 14px;
    }

    .sidebar-panel,
    .chat-panel {
        padding: 16px !important;
    }
}
"""

with gr.Blocks(
    title="RAG Document Assistant",
) as demo:

    gr.Markdown(
        """
        # Knowledge Assistant
        Ask questions about your documents in a focused, conversational workspace.
        """,
        elem_classes=["app-header"],
    )

    with gr.Row(elem_classes=["workspace-layout"]):

        # ------------------------------------------
        # Left panel: document management
        # ------------------------------------------

        with gr.Column(scale=1, min_width=290, elem_classes=["sidebar-panel"]):

            gr.Markdown(
                """
                ### Knowledge base
                Add documents to give your assistant context for its answers.
                """
            )

            files = gr.File(
                label="Documents",
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
                label="When indexing",
            )

            index_button = gr.Button(
                "Add documents",
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

        # ------------------------------------------
        # Right panel: provider selection and chat
        # ------------------------------------------

        with gr.Column(scale=3, min_width=420, elem_classes=["chat-panel"]):

            gr.Markdown("### Assistant settings")

            with gr.Row():
                provider_dropdown = gr.Dropdown(
                    label="Provider",
                    choices=PROVIDERS,
                    value="Ollama" if "Ollama" in PROVIDERS else None,
                    interactive=True,
                    scale=1,
                )

                api_key_input = gr.Textbox(
                    label="API key",
                    placeholder="Enter your provider API key",
                    type="password",
                    value="",
                    visible=(
                        bool(PROVIDERS)
                        and provider_dropdown.value != "Ollama"
                    ),
                    scale=2,
                )

            provider_status = gr.Markdown(
                "Select a provider and refresh its available models."
            )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=None,
                    interactive=True,
                    scale=4,
                )

                refresh_button = gr.Button(
                    "Refresh models",
                    scale=1,
                )

            # Load models on application startup.
            demo.load(
                fn=refresh_model_choices,
                inputs=[provider_dropdown, api_key_input],
                outputs=[model_dropdown, provider_status],
            )

            # Refresh models when the user clicks the button.
            refresh_button.click(
                fn=refresh_model_choices,
                inputs=[provider_dropdown, api_key_input],
                outputs=[model_dropdown, provider_status],
            )

            # Update the key field and clear stale model choices
            # when the selected provider changes.
            provider_dropdown.change(
                fn=update_api_key_visibility,
                inputs=[provider_dropdown],
                outputs=[
                    api_key_input,
                    provider_status,
                    model_dropdown,
                ],
            )

            # Refresh models when the user submits the API key.
            api_key_input.submit(
                fn=refresh_model_choices,
                inputs=[provider_dropdown, api_key_input],
                outputs=[model_dropdown, provider_status],
            )

            gr.Markdown("### Conversation")

            gr.ChatInterface(
                fn=chat,
                additional_inputs=[
                    provider_dropdown,
                    model_dropdown,
                    api_key_input,
                ],
                chatbot=gr.Chatbot(
                    height=560,
                    show_label=False,
                    layout="bubble",
                    placeholder="Ask a question about your indexed documents.",
                ),
                title="Ask your documents",
                description=(
                    "Ask a question, or choose a suggestion to get started."
                ),
                examples=[
                    [
                        "Summarize my uploaded documents.",
                        "Ollama",
                        None,
                        "",
                    ],
                    [
                        "What are the most important points?",
                        "Ollama",
                        None,
                        "",
                    ],
                    [
                        "Which page contains this information?",
                        "Ollama",
                        None,
                        "",
                    ],
                ],
            )


if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )