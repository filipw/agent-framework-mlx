# Agent Framework MLX

This library provides an [Apple MLX LM](https://github.com/ml-explore/mlx-lm) backend for the [Agent Framework](https://github.com/microsoft/agent-framework). It allows you to run Language Models locally on macOS with Apple Silicon using the `mlx-lm` library.

## Features

- **Local Inference**: Run models directly on your Mac using Apple Silicon
- **Tool Calling**: Native support for agentic tool invocation.
- **Observability**: Built-in OpenTelemetry tracing, metrics, and token usage tracking.
- **Streaming Support**: Full support for streaming responses.
- **Configurable Generation**: Fine-tune generation parameters like temperature, top-p, repetition penalty, and more.
- **Message Preprocessing**: Hook into the pipeline to modify messages before they are converted to prompts.
- **Agent Framework Integration**: Seamlessly plugs into the Agent Framework's `BaseChatClient` interface.

## Installation

Ensure you have Python 3.9+ and are running on macOS with Apple Silicon.

```bash
pip install agent-framework-mlx
```

Or install from source:

```bash
git clone https://github.com/filipw/agent-framework-mlx.git
cd agent-framework-mlx
pip install -e .
```

## Usage

### Basic Example

```python
import asyncio
from agent_framework import ChatMessage, Role, ChatOptions
from agent_framework_mlx import MLXChatClient, MLXGenerationConfig

# Initialize the client
client = MLXChatClient(
    model_path="mlx-community/Phi-4-mini-instruct-4bit",
    generation_config=MLXGenerationConfig(
        temp=0.7,
        max_tokens=500
    )
)

# Create messages
messages = [
    ChatMessage(role=Role.SYSTEM, text="You are a helpful assistant."),
    ChatMessage(role=Role.USER, text="Why is the sky blue?")
]

# Get response
response = await client.get_response(messages=messages)
print(response.text)
```

### Agent with Tools

You can easily create agents capable of calling tools (Python functions) using the client:

```python
from typing import Annotated

def calculate_bmi(weight: float, height: float) -> str:
    """Calculates BMI."""
    return f"{weight / (height ** 2):.2f}"

# Create an agent with the tool
agent = client.as_agent(
    name="HealthAssistant",
    instructions="You are a helpful assistant.",
    tools=[calculate_bmi]
)

response = await agent.run("Calculate BMI for 70kg and 1.75m")
print(response)
```

### Workflow Integration

You can use the client as backbone for Agent Framework agents when building agentic workflows:

```python
from agent_framework import ChatAgent

# notice the client constructed in the previous example now backs the local agent
local_agent = client.as_agent(
    name="Local_MLX",
    instructions="You are a helpful assistant."
)

remote_agent = ChatAgent(
    name="Cloud_LLM",
    instructions="You are a fallback expert. The previous assistant was unsure. Provide a complete answer.",
    chat_client=azure_client
)

builder = WorkflowBuilder()
builder.set_start_executor(local_agent)

builder.add_edge(
    source=local_agent,
    target=remote_agent,
    condition=should_fallback_to_cloud
)

workflow = builder.build()
```

### Streaming

```python
async for update in client.get_streaming_response(messages=messages):
    print(update.text, end="", flush=True)
```

### Configuration

You can configure the client using environment variables or a `.env` file. Using environment variables like `MLX_MODEL_PATH` allows you to omit arguments in code.

```bash
export MLX_MODEL_PATH="mlx-community/Phi-4-mini-instruct-4bit"
```

```python
# No arguments needed if env vars are set
client = MLXChatClient()
```

### Advanced Configuration

You can configure generation parameters globally via `MLXGenerationConfig` or per-request via `MLXChatOptions`.

```python
config = MLXGenerationConfig(
    temp=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    seed=42
)
```

### Message Preprocessing

You can intercept and modify messages before they are sent to the model. This is useful for injecting instructions or formatting content.

```python
def inject_instruction(messages):
    if messages:
        messages[-1]["content"] += "\nIMPORTANT: Be concise."
    return messages

client = MLXChatClient(
    model_path="...",
    message_preprocessor=inject_instruction
)
```

## Requirements

- macOS
- Apple Silicon
- Python 3.9+

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.