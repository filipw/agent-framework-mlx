from typing import Any, List, Optional, Union, ClassVar, TypedDict, Generic, TypeVar, AsyncIterable
from pydantic import BaseModel, ConfigDict

TOptions_co = TypeVar("TOptions_co", bound=TypedDict, covariant=True)

TOptions = TypeVar("TOptions", bound=TypedDict)

class ChatMiddlewareLayer(Generic[TOptions]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FunctionInvocationLayer(Generic[TOptions]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ChatTelemetryLayer(Generic[TOptions]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ResponseStream(Generic[TOptions_co]):
    """Minimal mock for ResponseStream."""
    def __init__(self, stream: AsyncIterable, *, finalizer=None, **kwargs):
        self._stream = stream
        self._finalizer = finalizer
        self._collected: List[Any] = []

    def __aiter__(self):
        return self._collecting_iter()

    async def _collecting_iter(self):
        async for item in self._stream:
            self._collected.append(item)
            yield item

    def __await__(self):
        async def _noop():
            pass
        return _noop().__await__()

    async def get_final_response(self):
        if self._finalizer is not None:
            return self._finalizer(self._collected)
        return self._collected


Role = str

class Content(BaseModel):
    type: str
    text: Optional[str] = None
    usage_details: Optional[dict[str, Any]] = None

    @classmethod
    def from_text(cls, text: str):
        return cls(type="text", text=text)
    
    @classmethod
    def from_usage(cls, usage_details: dict[str, Any]):
        return cls(type="usage", usage_details=usage_details)

class UsageDetails(dict):
    pass

class Message:
    """A plain python class to mock the Framework's non-Pydantic Message."""
    def __init__(self, role: Union[Role, str], contents: List[Any] = None, text: str = None):
        if isinstance(role, dict):
             # Handle possible dict role
             role = Role(role.get("value", "user"))
        elif isinstance(role, str):
            role = Role(role)
        self.role = role
        self.contents = contents or []
        if text:
            self.contents.append(Content.from_text(text=text))
    
    @property
    def text(self):
        return "".join([c.text for c in self.contents if c.type == "text" and c.text is not None])

class ChatOptions(TypedDict, total=False):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    seed: Optional[int]
    min_p: Optional[float]
    top_k: Optional[int]
    xtc_probability: Optional[float]
    xtc_threshold: Optional[float]
    repetition_penalty: Optional[float]
    repetition_context_size: Optional[int]

class ChatResponse(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    usage_details: Optional[dict[str, Any]] = None
    conversation_id: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_updates(cls, updates: List[Any], **kwargs) -> "ChatResponse":
        text = "".join(
            c.text for u in updates
            for c in (u.contents or [])
            if hasattr(c, "type") and c.type == "text" and c.text is not None
        )
        usage = next(
            (c.usage_details for u in updates for c in (u.contents or [])
             if hasattr(c, "type") and c.type == "usage"),
            None,
        )
        model = next((u.model for u in reversed(updates) if getattr(u, "model", None)), None)
        content = Content.from_text(text=text)
        message = Message("assistant", [content])
        return cls(messages=[message], model=model, usage_details=usage)

class ChatResponseUpdate(BaseModel):
    role: Optional[Union[Role, str]] = None
    contents: List[Any]
    model: Optional[str] = None
    
    @property
    def text(self):
        return "".join([c.text for c in self.contents if c.type == "text" and c.text is not None])

    model_config = ConfigDict(arbitrary_types_allowed=True)

class BaseChatClient(Generic[TOptions_co]):
    def __init__(self, **kwargs):
        pass
    
    async def get_response(self, *args, **kwargs):
        pass

def load_settings(settings_type, *, env_prefix="", env_file_path=None, env_file_encoding=None, required_fields=None, **overrides):
    return {k: v for k, v in overrides.items() if v is not None}


class IntegrationInitializationError(Exception):
    pass

Contents = Union[Content]