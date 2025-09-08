"""
Simple text‑based test harness for the Ollama chatbot, without audio or MQTT.

This script allows you to interact with your local Ollama model by typing
messages.  It uses the same environment variables as the full voice bot
to configure the model and system prompt.  The conversation history is
maintained across user inputs so the model has context.  To end the
conversation, type ``exit`` or press Ctrl‑C.

Usage:
  uv run python test_chat.py

Required environment variables (can be set in a .env file):

  OLLAMA_BASE_URL  - base URL of your local Ollama server (default: http://127.0.0.1:11434)
  OLLAMA_MODEL     - name of the model to use (e.g. llama3:8b-instruct)
  SYSTEM_PROMPT    - optional system prompt to guide the model

Example .env:

    OLLAMA_BASE_URL=http://127.0.0.1:11434
    OLLAMA_MODEL=llama3:8b-instruct
    SYSTEM_PROMPT=You are a concise, helpful assistant.

When run, the script will prompt you for input (``You:``).  It will send
your input along with the conversation history to Ollama and print the
model's reply.  This provides a quick way to verify that your Ollama
server is running and that your network and API configuration are
correct.
"""

import os
import requests
from dotenv import load_dotenv


def main() -> None:
    # Load environment variables from .env if present
    load_dotenv()

    # Read configuration
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct")
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a concise, helpful assistant. Answer succinctly and politely.",
    )

    # Initialize conversation history with system prompt
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    print("\nText-based Ollama chatbot test.")
    print("Type 'exit' or press Ctrl-C to quit.\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat.")
                break
            # Append user message to history
            messages.append({"role": "user", "content": user_input})

            # Call Ollama chat API
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            try:
                resp = requests.post(
                    f"{base_url}/api/chat", json=payload, timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data.get("message", {}).get("content", "").strip()
            except Exception as exc:
                reply = f"[error] Ollama API error: {exc}"

            # Print the reply and append to history
            print(f"Bot: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting chat.")


if __name__ == "__main__":
    main()


