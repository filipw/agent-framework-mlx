from typing import Any, List, Optional, Union, ClassVar
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

class BaseContent(BaseModel):
    pass

class TextContent(BaseContent):
    text: str
    def __init__(self, text: str, **kwargs):
        super().__init__(text=text, **kwargs)


class UsageDetails(BaseModel):
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    total_token_count: Optional[int] = None

class UsageContent(BaseContent):
    details: UsageDetails
    def __init__(self, details: UsageDetails, **kwargs):
        super().__init__(details=details, **kwargs)

class ChatMessage:
    """A plain python class to mock the Framework's non-Pydantic ChatMessage."""
    def __init__(self, role: Union[Role, str], contents: List[Any] = None, text: str = None):
        self.role = role
        self.contents = contents or []
        if text:
            self.contents.append(TextContent(text=text))
    
    @property
    def text(self):
        return "".join([c.text for c in self.contents if isinstance(c, TextContent)])

class ChatOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    additional_properties: Optional[dict] = {}

class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    model_id: str
    usage_details: Optional[UsageDetails] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ChatResponseUpdate(BaseModel):
    role: Union[Role, str] 
    contents: List[Any]
    model_id: str
    
    @property
    def text(self):
        return "".join([c.text for c in self.contents if isinstance(c, TextContent)])

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

Contents = Union[TextContent, BaseContent, UsageContent]