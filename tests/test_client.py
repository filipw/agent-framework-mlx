import pytest
import json
from unittest.mock import MagicMock, patch
from agent_framework import Message, ChatOptions
from agent_framework.exceptions import IntegrationInitializationError
import agent_framework_mlx.client
from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
from agent_framework_mlx.client import MLXChatOptions, _tool_to_function_spec, _extract_tool_calls

@pytest.mark.asyncio
async def test_client_initialization(mock_mlx):
    config = MLXGenerationConfig(temp=0.5, max_tokens=500)
    client = MLXChatClient(model_path="test/model", generation_config=config)
    
    assert client.generation_config.temp == 0.5
    assert client.model_id == "test/model"

@pytest.mark.asyncio
async def test_client_init_no_tokenizer(mock_mlx):
    agent_framework_mlx.client.load.return_value = (MagicMock(), None)
    
    with pytest.raises(IntegrationInitializationError, match="Failed to load tokenizer"):
        MLXChatClient(model_path="test/model")

@pytest.mark.asyncio
async def test_sampler_configuration(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    options = MLXChatOptions(
        temperature=0.7,
        top_p=0.9,
        min_p=0.05,
        top_k=50,
        xtc_probability=0.1,
        xtc_threshold=0.5
    )
    
    client._get_sampler(options)
    
    agent_framework_mlx.client.make_sampler.assert_called_with(
        temp=0.7,
        top_p=0.9,
        min_p=0.05,
        min_tokens_to_keep=1, # default
        top_k=50,
        xtc_probability=0.1,
        xtc_threshold=0.5
    )

@pytest.mark.asyncio
async def test_logits_processors_configuration(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    options = MLXChatOptions(
        repetition_penalty=1.2,
        repetition_context_size=50
    )
    
    client._get_logits_processors(options)
    
    agent_framework_mlx.client.make_logits_processors.assert_called_with(
        repetition_penalty=1.2,
        repetition_context_size=50
    )

@pytest.mark.asyncio
async def test_seed_parameter(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    options = MLXChatOptions(seed=42)
    messages = [Message(role="user", text="Hi")]

    await client._inner_get_response(messages=messages, options=options)
    
    # Check generate call kwargs
    args, kwargs = agent_framework_mlx.client.generate.call_args
    assert kwargs["seed"] == 42

@pytest.mark.asyncio
async def test_streaming_error_propagation(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="Hi")]
    
    # Mock stream_generate to raise an exception
    def mock_stream_error(*args, **kwargs):
        raise RuntimeError("Generation failed")
        yield
        
    with patch("agent_framework_mlx.client.stream_generate", side_effect=mock_stream_error):
        with pytest.raises(RuntimeError, match="Generation failed"):
            async for _ in client._inner_get_response(
                messages=messages,
                stream=True,
                options=MLXChatOptions()
            ):
                pass

@pytest.mark.asyncio
async def test_prepare_prompt_fallback(mock_mlx):
    # Setup tokenizer without apply_chat_template
    # We can just create a client and swap the tokenizer
    client = MLXChatClient(model_path="test/model")
    client.mlx_tokenizer = MagicMock(spec=[])
    
    messages = [Message(role="user", text="Hi")]
    prompt = client._prepare_prompt(messages)
    
    assert prompt == "user: Hi"

@pytest.mark.asyncio
async def test_instructions_prepended_as_system_message(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="Hi")]
    options = MLXChatOptions(instructions="You are a helpful health assistant.")

    await client._inner_get_response(messages=messages, options=options)

    call_args = client.mlx_tokenizer.apply_chat_template.call_args #type: ignore
    passed_msgs = call_args[0][0]
    assert passed_msgs[0]["role"] == "system"
    assert passed_msgs[0]["content"] == "You are a helpful health assistant."

@pytest.mark.asyncio
async def test_message_preprocessor(mock_mlx):
    def add_instruction(messages):
        if messages:
            messages[-1]["content"] += " [INSTRUCTION]"
        return messages

    client = MLXChatClient(model_path="test/model", message_preprocessor=add_instruction)
    messages = [Message(role="user", text="Hi")]
    
    await client._inner_get_response(messages=messages, options=MLXChatOptions())
    
    call_args = client.mlx_tokenizer.apply_chat_template.call_args #type: ignore
    assert call_args is not None
    passed_msgs = call_args[0][0]
    assert passed_msgs[0]["content"] == "Hi [INSTRUCTION]"

@pytest.mark.asyncio
async def test_get_response(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="Hi")]
    
    response = await client._inner_get_response(
        messages=messages, 
        options=MLXChatOptions()
    )
    
    assert response.messages[0].contents[0].text == "Mock Output" #type: ignore
    assert response.model == "test/model"

@pytest.mark.asyncio
async def test_streaming_response(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="Hi")]
    
    response_text = ""
    async for update in client._inner_get_response(
        messages=messages,
        stream=True,
        options=MLXChatOptions()
    ):
        response_text += update.text
        
    assert response_text == "Mock Chunk"

@pytest.mark.asyncio
async def test_hierarchical_configuration(mock_mlx):
    # setup client with custom config
    config = MLXGenerationConfig(temp=0.9, max_tokens=123, seed=99)
    client = MLXChatClient(model_path="test/model", generation_config=config)
    messages = [Message(role="user", text="Hi")]
    
    # 1. test fallback to config when options are empty
    await client._inner_get_response(messages=messages, options=MLXChatOptions())
    
    # verify make_sampler was called with config values
    agent_framework_mlx.client.make_sampler.assert_called()
    sampler_args = agent_framework_mlx.client.make_sampler.call_args[1]
    assert sampler_args["temp"] == 0.9
    
    # verify generate was called with config values
    args, kwargs = agent_framework_mlx.client.generate.call_args
    assert kwargs["max_tokens"] == 123
    assert kwargs["seed"] == 99

    # 2. test override of config when options are provided
    override_options = MLXChatOptions(temperature=0.1, max_tokens=456, seed=1)
    await client._inner_get_response(messages=messages, options=override_options)
    
    sampler_args_override = agent_framework_mlx.client.make_sampler.call_args[1]
    assert sampler_args_override["temp"] == 0.1
    
    args_override, kwargs_override = agent_framework_mlx.client.generate.call_args
    assert kwargs_override["max_tokens"] == 456
    assert kwargs_override["seed"] == 1


# --- Tool calling ---

def test_tool_to_function_spec_from_dict():
    tool = {
        "name": "calculate_bmi",
        "description": "Calculates BMI",
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Weight in kg"},
                "height_m": {"type": "number", "description": "Height in meters"},
                "unit": {"type": "string", "description": "Unit system", "default": "metric"},
            },
            "required": ["weight_kg", "height_m"],
        },
    }

    spec = _tool_to_function_spec(tool)

    assert spec == {
        "name": "calculate_bmi",
        "description": "Calculates BMI",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kg"},
            "height_m": {"type": "number", "description": "Height in meters"},
            "unit": {"type": "string", "description": "Unit system", "default": "metric"},
        },
    }

def test_tool_to_function_spec_from_openai_style_dict():
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }

    spec = _tool_to_function_spec(tool)

    assert spec == {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"city": {"type": "string", "description": "City name"}},
    }

def test_tool_to_function_spec_from_object():
    tool = MagicMock()
    tool.name = "get_weather"
    tool.description = "Get the weather"
    tool.parameters.return_value = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    spec = _tool_to_function_spec(tool)

    assert spec == {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"city": {"type": "string"}},
    }

def test_tool_to_function_spec_unknown_shape_returns_none():
    assert _tool_to_function_spec({"description": "no name here"}) is None

def test_extract_tool_calls_plain_json():
    text = '[{"name": "calculate_bmi", "arguments": {"weight_kg": 70, "height_m": 1.75}}]'
    assert _extract_tool_calls(text) == [
        {"name": "calculate_bmi", "arguments": {"weight_kg": 70, "height_m": 1.75}}
    ]

def test_extract_tool_calls_with_surrounding_text():
    text = 'Sure, here it is: [{"name": "calculate_bmi", "arguments": {"weight_kg": 70}}] done.'
    assert _extract_tool_calls(text) == [{"name": "calculate_bmi", "arguments": {"weight_kg": 70}}]

def test_extract_tool_calls_returns_none_for_plain_text():
    assert _extract_tool_calls("Your BMI is approximately 22.86.") is None

def test_extract_tool_calls_handles_tool_call_tags_and_trailing_hallucination():
    # Real-world Phi output: tagged tool call followed by a fabricated follow-up turn.
    text = (
        '<|tool_call|>[{"name": "get_weather_updates", "arguments": {"city": "Paris"}}]'
        '<|/tool_call|><|user|>How do I use this? Here is an example: [1, 2, 3]'
    )
    assert _extract_tool_calls(text) == [
        {"name": "get_weather_updates", "arguments": {"city": "Paris"}}
    ]

@pytest.mark.asyncio
async def test_prepare_prompt_injects_tools_into_system_message(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="system", text="You are helpful."), Message(role="user", text="Hi")]
    tool_specs = [{"name": "calculate_bmi", "description": "d", "parameters": {}}]

    client._prepare_prompt(messages, tool_specs)

    call_args = client.mlx_tokenizer.apply_chat_template.call_args #type: ignore
    passed_msgs = call_args[0][0]
    assert passed_msgs[0]["role"] == "system"
    assert json.loads(passed_msgs[0]["tools"]) == tool_specs

@pytest.mark.asyncio
async def test_prepare_prompt_creates_system_message_for_tools_if_missing(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="Hi")]
    tool_specs = [{"name": "calculate_bmi", "description": "d", "parameters": {}}]

    client._prepare_prompt(messages, tool_specs)

    call_args = client.mlx_tokenizer.apply_chat_template.call_args #type: ignore
    passed_msgs = call_args[0][0]
    assert passed_msgs[0]["role"] == "system"
    assert json.loads(passed_msgs[0]["tools"]) == tool_specs

@pytest.mark.asyncio
async def test_tool_call_detected_in_response(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="What is my BMI?")]
    tools = [{
        "name": "calculate_bmi",
        "description": "Calculates BMI",
        "parameters": {
            "type": "object",
            "properties": {"weight_kg": {"type": "number"}, "height_m": {"type": "number"}},
            "required": ["weight_kg", "height_m"],
        },
    }]
    options = MLXChatOptions(tools=tools)

    agent_framework_mlx.client.generate.return_value = (
        '[{"name": "calculate_bmi", "arguments": {"weight_kg": 70, "height_m": 1.75}}]'
    )

    response = await client._inner_get_response(messages=messages, options=options)

    result_content = response.messages[0].contents[0]
    assert result_content.type == "function_call"
    assert result_content.name == "calculate_bmi"
    assert result_content.arguments == {"weight_kg": 70, "height_m": 1.75}

@pytest.mark.asyncio
async def test_no_tool_call_falls_back_to_text(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="What is my BMI?")]
    tools = [{"name": "calculate_bmi", "description": "d", "parameters": {}}]
    options = MLXChatOptions(tools=tools)

    agent_framework_mlx.client.generate.return_value = "Your BMI is approximately 22.86."

    response = await client._inner_get_response(messages=messages, options=options)

    result_content = response.messages[0].contents[0]
    assert result_content.type == "text"
    assert result_content.text == "Your BMI is approximately 22.86."

@pytest.mark.asyncio
async def test_streaming_tool_call_yields_function_call_update(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [Message(role="user", text="What is my BMI?")]
    tools = [{"name": "calculate_bmi", "description": "d", "parameters": {}}]
    options = MLXChatOptions(tools=tools)

    tool_call_json = '[{"name": "calculate_bmi", "arguments": {"weight_kg": 70, "height_m": 1.75}}]'

    def mock_stream(*args, **kwargs):
        class Chunk:
            def __init__(self, text):
                self.text = text
        for piece in [tool_call_json[:10], tool_call_json[10:]]:
            yield Chunk(piece)

    with patch("agent_framework_mlx.client.stream_generate", side_effect=mock_stream):
        updates = [
            update async for update in client._inner_get_response(messages=messages, stream=True, options=options)
        ]

    function_call_updates = [u for u in updates for c in u.contents if c.type == "function_call"]
    assert len(function_call_updates) == 1
    call_content = function_call_updates[0].contents[0]
    assert call_content.name == "calculate_bmi"
    assert call_content.arguments == {"weight_kg": 70, "height_m": 1.75}
    # raw JSON chunks must not leak out as text updates
    assert not any(c.type == "text" for u in updates for c in u.contents)
