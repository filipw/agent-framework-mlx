from typing import Any, List, Optional, Union, ClassVar, TypedDict
from pydantic import BaseModel, ConfigDict

def use_chat_middleware(cls):
    return cls

def use_function_invocation(cls):
    return cls

def use_instrumentation(cls):
    return cls

class Role:
    value: str
    def __init__(self, value: str):
        self.value = value
    
    SYSTEM: "Role"
    USER: "Role"
    ASSISTANT: "Role"
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return isinstance(other, Role) and self.value == other.value

    def __str__(self):
        return self.value

Role.SYSTEM = Role("system")
Role.USER = Role("user")
Role.ASSISTANT = Role("assistant")

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

class ChatMessage:
    """A plain python class to mock the Framework's non-Pydantic ChatMessage."""
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
    messages: List[ChatMessage]
    model_id: str
    usage_details: Optional[dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ChatResponseUpdate(BaseModel):
    role: Optional[Union[Role, str]] = None
    contents: List[Any]
    model_id: str
    
    @property
    def text(self):
        return "".join([c.text for c in self.contents if c.type == "text" and c.text is not None])

    model_config = ConfigDict(arbitrary_types_allowed=True)

class BaseChatClient:
    def __init__(self, **kwargs):
        pass
    
    async def get_response(self, *args, **kwargs):
        pass

class AFBaseSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    env_prefix: ClassVar[str] = ""


class ServiceInitializationError(Exception):
    pass

Contents = Union[Content]