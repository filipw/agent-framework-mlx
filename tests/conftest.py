import sys
import os
import pytest
from unittest.mock import MagicMock

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import mock_framework

sys.modules["agent_framework"] = mock_framework
sys.modules["agent_framework.observability"] = mock_framework
sys.modules["agent_framework._pydantic"] = mock_framework
sys.modules["agent_framework.exceptions"] = mock_framework

@pytest.fixture
def mock_mlx():
    """Mocks the heavy MLX libraries by patching the client's namespace."""
    with pytest.MonkeyPatch.context() as m:
        mock_load = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Mock Prompt"
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        m.setattr("agent_framework_mlx.client.load", mock_load)
        
        mock_generate = MagicMock(return_value="Mock Output")
        m.setattr("agent_framework_mlx.client.generate", mock_generate)
        
        def mock_stream(*args, **kwargs):
            class Chunk:
                text = "Mock Chunk"
            yield Chunk()
            
        m.setattr("agent_framework_mlx.client.stream_generate", mock_stream)
        m.setattr("agent_framework_mlx.client.make_sampler", MagicMock())
        m.setattr("agent_framework_mlx.client.make_logits_processors", MagicMock())

        yield