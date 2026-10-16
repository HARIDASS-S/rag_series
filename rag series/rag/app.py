
import gradio as gr

from rag_engine import answer_question


def respond(
    message,
    history,
    department,
    year,
    category
):
    # Gradio's message-format history uses role/content dictionaries.
    history = history or []

    normalized_history = []

    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")

            # Ignore non-text multimodal history entries.
            if isinstance(content, str) and role in {
                "user", "assistant"
            }:
                normalized_history.append({
                    "role": role,
                    "content": content
                })

    result = answer_question(
        question=message,
        history=normalized_history,
        department=department or None,
        year=year or None,
        category=category or None
    )

    answer = result["answer"]

    if result["sources"]:
        answer += "\n\n### Retrieved sources\n"

        for source in result["sources"]:
            answer += (
                f"\n- **{source['source']}**"
                f"\n\n  > {source['text'][:350].replace(chr(10), ' ')}"
                "\n"
            )

    return answer


with gr.Blocks(title="College RAG Assistant") as demo:
    gr.Markdown(
        """
        # 🎓 College RAG Assistant

        Ask questions about college documents using a local AI model.
        Answers are grounded in indexed documents and include source references.
        """
    )

    with gr.Row():
        department = gr.Textbox(
            label="Department filter",
            placeholder="e.g. AIDS"
        )

        year = gr.Textbox(
            label="Year filter",
            placeholder="e.g. 2026"
        )

        category = gr.Textbox(
            label="Category filter",
            placeholder="e.g. placement"
        )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[
            department,
            year,
            category
        ],
        examples=[
            ["What courses are available?", "", "", ""],
            ["What placement opportunities are listed?", "", "", ""],
            ["Which page contains the attendance policy?", "", "", ""]
        ]
    )


if __name__ == "__main__":
    demo.launch()