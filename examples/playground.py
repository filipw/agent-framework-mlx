import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from agent_framework import ChatMessage, Role, ChatOptions
    from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
except ImportError as e:
    print("Error: Dependencies not found.")
    print("Make sure you run: pip install -e .")
    print(f"Details: {e}")
    sys.exit(1)

async def main():
    model_path = "mlx-community/Phi-4-mini-instruct-4bit"
    
    print(f"--- 🚀 Loading Model: {model_path} ---")
    
    config = MLXGenerationConfig(
        temp=0.7,
        max_tokens=200,
        verbose=False
    )

    client = MLXChatClient(
        model_path=model_path,
        generation_config=config
    )

    prompt_text = "Explain quantum computing to a 5 year old in one sentence."
    print(f"\n📝 User: {prompt_text}\n")
    
    messages = [
        ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant."),
        ChatMessage(role=Role.USER, text=prompt_text)
    ]
    options = ChatOptions()

    print("--- ⚡️ Running Standard Generation ---")
    response = await client.get_response(messages=messages, chat_options=options)
    print(f"🤖 Assistant: {response.text}")

    print("\n--- 🌊 Running Streaming Generation ---")
    print("🤖 Assistant: ", end="", flush=True)
    
    async for update in client.get_streaming_response(messages=messages, chat_options=options):
        print(update.text, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())