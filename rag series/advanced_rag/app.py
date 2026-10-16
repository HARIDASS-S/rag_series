

import gradio as gr
import ollama

from rag import ask_question



def rewrite_question(question, history):
    
    if not history:
        return question

    history_texts = []
    for item in history[-4:]: # last 4 messages
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            history_texts.append(f"{role}: {content}")
        elif isinstance(item, (list, tuple)):
            # old format (user, bot)
            history_texts.append(f"user: {item[0]}\nassistant: {item[1]}")
        else:
            history_texts.append(str(item))

    conversation_context = "\n".join(history_texts)


    prompt = f"""
Rewrite the latest question
as a standalone question.

Conversation:

{ conversation_context }

Latest question:

{ question }

Return ONLY the rewritten question.
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def generate_multi_queries(question):
    """Generates 3 different versions of the question to improve retrieval."""
    prompt = f"""
    Generate 3 different search queries for the following question to help find relevant documents in a vector database.
    Question: {question}
    Return one query per line. Do not include numbers or bullet points.
    """
    
    response = ollama.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    queries = response["message"]["content"].strip().splitlines()
    # Clean up any empty lines or numbering if the LLM added them
    return [q.strip() for q in queries if q.strip()][:3]

def chat(message, history):
    if not message or not message.strip():
        return "Please enter a question."

    try:
        

        # 1. If there is history, rewrite the question to be standalone
        if history:
            print(f"Original: {message}")
            query_to_ask = rewrite_question(message, history)
            print(f"Rewritten: {query_to_ask}")
        else:
            query_to_ask = message

        # 2. Pass the (potentially rewritten) question to the RAG system
        answer = ask_question(query_to_ask)
        return answer

    except Exception as e:
        print("\n========== ERROR ==========")
        print(e)
        print("===========================\n")
        return f"❌ Error: {str(e)}"

demo = gr.ChatInterface(
    fn=chat,
    title="🎓 College RAG Assistant",
    description=(
        "Ask questions about the college using "
        "your uploaded documents.\n\n"
        "Powered by Ollama + ChromaDB + RAG"
    ),
    textbox=gr.Textbox(
        placeholder="Example: What is the name of the college?",
        label="Ask your question"
    ),
    examples=[
        "What is the name of the college?",
        "What courses are available?",
        "What is the placement information?",
        "What facilities are available?",
    ],
)


if __name__ == "__main__":
   
    demo.launch()