import pytest
from unittest.mock import MagicMock, patch
from agent_framework import ChatMessage, Role, ChatOptions
from agent_framework.exceptions import ServiceInitializationError
import agent_framework_mlx.client
from agent_framework_mlx import MLXChatClient, MLXGenerationConfig
from agent_framework_mlx.client import MLXChatOptions

@pytest.mark.asyncio
async def test_client_initialization(mock_mlx):
    config = MLXGenerationConfig(temp=0.5, max_tokens=500)
    client = MLXChatClient(model_path="test/model", generation_config=config)
    
    assert client.generation_config.temp == 0.5
    assert client.model_id == "test/model"

@pytest.mark.asyncio
async def test_client_init_no_tokenizer(mock_mlx):
    agent_framework_mlx.client.load.return_value = (MagicMock(), None)
    
    with pytest.raises(ServiceInitializationError, match="Failed to load tokenizer"):
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
    messages = [ChatMessage(role=Role.USER, text="Hi")]

    await client._inner_get_response(messages=messages, options=options)
    
    # Check generate call kwargs
    args, kwargs = agent_framework_mlx.client.generate.call_args
    assert kwargs["seed"] == 42

@pytest.mark.asyncio
async def test_streaming_error_propagation(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [ChatMessage(role=Role.USER, text="Hi")]
    
    # Mock stream_generate to raise an exception
    def mock_stream_error(*args, **kwargs):
        raise RuntimeError("Generation failed")
        yield
        
    with patch("agent_framework_mlx.client.stream_generate", side_effect=mock_stream_error):
        with pytest.raises(RuntimeError, match="Generation failed"):
            async for _ in client._inner_get_streaming_response(
                messages=messages, 
                options=MLXChatOptions()
            ):
                pass

@pytest.mark.asyncio
async def test_prepare_prompt_fallback(mock_mlx):
    # Setup tokenizer without apply_chat_template
    # We can just create a client and swap the tokenizer
    client = MLXChatClient(model_path="test/model")
    client.tokenizer = MagicMock(spec=[])
    
    messages = [ChatMessage(role=Role.USER, text="Hi")]
    prompt = client._prepare_prompt(messages)
    
    assert prompt == "user: Hi"

@pytest.mark.asyncio
async def test_message_preprocessor(mock_mlx):
    def add_instruction(messages):
        if messages:
            messages[-1]["content"] += " [INSTRUCTION]"
        return messages

    client = MLXChatClient(model_path="test/model", message_preprocessor=add_instruction)
    messages = [ChatMessage(role=Role.USER, text="Hi")]
    
    await client._inner_get_response(messages=messages, options=MLXChatOptions())
    
    call_args = client.tokenizer.apply_chat_template.call_args #type: ignore
    assert call_args is not None
    passed_msgs = call_args[0][0]
    assert passed_msgs[0]["content"] == "Hi [INSTRUCTION]"

@pytest.mark.asyncio
async def test_get_response(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [ChatMessage(role=Role.USER, text="Hi")]
    
    response = await client._inner_get_response(
        messages=messages, 
        options=MLXChatOptions()
    )
    
    assert response.messages[0].contents[0].text == "Mock Output" #type: ignore
    assert response.model_id == "test/model"

@pytest.mark.asyncio
async def test_streaming_response(mock_mlx):
    client = MLXChatClient(model_path="test/model")
    messages = [ChatMessage(role=Role.USER, text="Hi")]
    
    response_text = ""
    async for update in client._inner_get_streaming_response(
        messages=messages, 
        options=MLXChatOptions()
    ):
        response_text += update.text
        
    assert response_text == "Mock Chunk"

@pytest.mark.asyncio
async def test_hierarchical_configuration(mock_mlx):
    # setup client with custom config
    config = MLXGenerationConfig(temp=0.9, max_tokens=123, seed=99)
    client = MLXChatClient(model_path="test/model", generation_config=config)
    messages = [ChatMessage(role=Role.USER, text="Hi")]
    
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
