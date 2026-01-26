import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
except ImportError as e:
    print("Error: Dependencies not found.")
    print("Make sure you run: pip install -e .")
    print(f"Details: {e}")
    sys.exit(1)

from typing import Annotated

def calculate_bmi(
    weight_kg: float, 
    height_m: float
) -> str:
    """Calculates the Body Mass Index (BMI) given weight in kg and height in meters."""
    bmi = weight_kg / (height_m ** 2)
    return f"{bmi:.2f}"

async def main():
    model_path = "mlx-community/Phi-4-mini-instruct-4bit"
    
    print(f"--- 🚀 Loading MLX Model: {model_path} ---")
    
    config = MLXGenerationConfig(
        temp=0.0, # Lower temp is better for tool use
        max_tokens=1000,
        verbose=False
    )

    client = MLXChatClient(
        model_path=model_path,
        generation_config=config,
    )
    
    # Create the agent
    agent = client.as_agent(
        name="HealthAssistant",
        instructions="You are a helpful health assistant. You can calculate BMI.",
        tools=[calculate_bmi]
    )

    prompt_text = "My weight is 70kg and my height is 1.75m. What is my BMI?"
    print(f"\n📝 User: {prompt_text}\n")
    
    print("--- ⚡️ Running Agent (Automatic Tool Use) ---")
    response = await agent.run(prompt_text)
    
    print(f"🤖 Assistant: {response}")

    print("\n--- 🌊 Running Streaming Agent ---")
    
    prompt_text_stream = "Now calculate it for someone who is 90kg and 1.80m."
    print(f"\n📝 User: {prompt_text_stream}\n")
    print("🤖 Assistant: ", end="", flush=True)
    
    async for update in agent.run_stream(prompt_text_stream):
        if update.text:
            print(update.text, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())