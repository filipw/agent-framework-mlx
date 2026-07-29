import asyncio
import logging
import functools
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterable, MutableSequence, Optional, Callable, TypedDict
from pydantic import BaseModel
from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    FunctionInvocationLayer,
    Message,
    ChatOptions, 
    ChatResponse, 
    ChatResponseUpdate,
    ResponseStream,
    prepend_instructions_to_messages,
    Content,
    UsageDetails,
    normalize_tools,
    load_settings,
)
from agent_framework.observability import ChatTelemetryLayer
from agent_framework.exceptions import IntegrationInitializationError
from mlx_lm.utils import load
from mlx_lm.generate import generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

logger = logging.getLogger(__name__)

_TOOL_CALL_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _tool_to_function_spec(tool: Any) -> Optional[dict[str, Any]]:
    """
    Converts an agent_framework tool (a ``FunctionTool``-like object exposing
    ``name``/``description``/``parameters()``, or a raw dict) into the flat
    function-calling schema that Phi-family models expect when embedded in the
    system message's ``<|tool|>...<|/tool|>`` block:

        {"name": ..., "description": ..., "parameters": {param: {"type", "description", "default"?}}}

    Returns None if the tool's shape isn't understood.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    schema: Optional[dict[str, Any]] = None

    if isinstance(tool, dict):
        if isinstance(tool.get("function"), dict):
            # OpenAI-style {"type": "function", "function": {...}}
            fn = tool["function"]
            name = fn.get("name")
            description = fn.get("description")
            schema = fn.get("parameters")
        else:
            name = tool.get("name")
            description = tool.get("description")
            schema = tool.get("parameters")
    else:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        parameters_fn = getattr(tool, "parameters", None)
        if callable(parameters_fn):
            try:
                candidate_schema = parameters_fn()
            except Exception:
                candidate_schema = None
            schema = candidate_schema if isinstance(candidate_schema, dict) else None

    if not name:
        return None

    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", []) if isinstance(schema, dict) else [])

    parameters: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        entry: dict[str, Any] = {"type": prop_schema.get("type", "string")}
        if "description" in prop_schema:
            entry["description"] = prop_schema["description"]
        if prop_name not in required and "default" in prop_schema:
            entry["default"] = prop_schema["default"]
        parameters[prop_name] = entry

    return {
        "name": name,
        "description": description or "",
        "parameters": parameters,
    }


_TOOL_CALL_TAG_RE = re.compile(r"<\|tool_call\|>(.*?)<\|/tool_call\|>", re.DOTALL)


def _find_first_json_array(text: str) -> Optional[str]:
    """
    Scans for the first balanced top-level JSON array in ``text`` (tracking bracket
    depth and string literals), ignoring anything before or after it. This is more
    robust than a greedy regex when the model keeps generating hallucinated content
    (e.g. a fabricated follow-up turn) after the actual tool call.
    """
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _extract_tool_calls(text: str) -> Optional[list[dict[str, Any]]]:
    """
    Best-effort parse of a Phi-style function-calling completion into a list of
    ``{"name": ..., "arguments": {...}}`` dicts. Handles both a bare JSON array and
    one wrapped in ``<|tool_call|>...<|/tool_call|>`` tags, optionally surrounded by
    other (possibly hallucinated) text. Returns None if no valid tool call could be found.
    """
    candidates: list[str] = []

    if match := _TOOL_CALL_TAG_RE.search(text):
        candidates.append(match.group(1).strip())

    candidates.append(text.strip())

    if array_text := _find_first_json_array(text):
        candidates.append(array_text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list) and parsed and all(isinstance(item, dict) and "name" in item for item in parsed):
            return parsed

    return None


class MLXGenerationConfig(BaseModel):
    """Configuration for MLX Model Generation defaults."""
    temp: float = 0.0
    top_p: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    max_tokens: int = 1000
    min_tokens_to_keep: int = 1
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = 20
    seed: Optional[int] = None
    verbose: bool = False

class MLXSettings(TypedDict, total=False):
    """
    MLX Client settings.
    
    Attributes:
        model_path: The path to the MLX model. (Env var MLX_MODEL_PATH)
        adapter_path: Optional path to an adapter. (Env var MLX_ADAPTER_PATH)
    """
    model_path: Optional[str]
    adapter_path: Optional[str]

class MLXChatOptions(ChatOptions, total=False):
    """MLX-specific Chat Options."""
    min_p: float
    top_k: int
    xtc_probability: float
    xtc_threshold: float
    repetition_penalty: float
    repetition_context_size: int

def _render_message_content(message: Message) -> str:
    """
    Renders a Message's content to a string for the chat template. Plain text content
    is used as-is. Messages that only carry function_call/function_result content (which
    ``Message.text`` ignores) are rendered so the model can see its own prior tool calls
    and their results in follow-up turns.
    """
    text = message.text if hasattr(message, "text") else ""
    if text:
        return text

    contents = list(getattr(message, "contents", None) or [])

    function_calls = [c for c in contents if getattr(c, "type", None) == "function_call"]
    if function_calls:
        calls: list[dict[str, Any]] = []
        for call in function_calls:
            arguments = call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
            calls.append({"name": call.name, "arguments": arguments})
        return f"<|tool_call|>{json.dumps(calls)}<|/tool_call|>"

    function_results = [c for c in contents if getattr(c, "type", None) == "function_result"]
    if function_results:
        return "\n".join(str(getattr(c, "result", "") or "") for c in function_results)

    return str(contents)


class MLXChatClient(ChatMiddlewareLayer[MLXChatOptions], FunctionInvocationLayer[MLXChatOptions], ChatTelemetryLayer[MLXChatOptions], BaseChatClient[MLXChatOptions]):
    """
    A Chat Client that runs models locally using Apple MLX.
    """
    OTEL_PROVIDER_NAME = "mlx_local"

    def __init__(
        self, 
        model_path: Optional[str] = None, 
        adapter_path: Optional[str] = None,
        tokenizer_config: Optional[dict] = None,
        generation_config: Optional[MLXGenerationConfig] = None,
        message_preprocessor: Optional[Callable[[list[dict[str, str]]], list[dict[str, str]]]] = None,
        env_file_path: Optional[str] = None,
        env_file_encoding: str = "utf-8",
        **kwargs: Any
    ):
        settings = load_settings(
            MLXSettings,
            env_prefix="MLX_",
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
            required_fields=["model_path"],
            model_path=model_path,
            adapter_path=adapter_path,
        )

        super().__init__(**kwargs)
        
        self.generation_config = generation_config or MLXGenerationConfig()
        self.message_preprocessor = message_preprocessor

        # MLX binds model weights and generation state to the thread they are created/used on.
        # A single dedicated worker thread ensures the model is always loaded from and run on
        # the same OS thread, avoiding "There is no Stream(...) in current thread" errors.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")

        logger.info(f"Loading MLX Model: {settings['model_path']}...")
        loaded = self._executor.submit(
            load,
            settings["model_path"],
            adapter_path=settings.get("adapter_path"),
            tokenizer_config=tokenizer_config or {}
        ).result() # type: ignore
        
        # Handle variable return length from mlx_lm.load
        if isinstance(loaded, tuple):
            self.model = loaded[0]
            self.mlx_tokenizer = loaded[1]
        else:
            self.model = loaded
            self.mlx_tokenizer = None 

        if self.mlx_tokenizer is None:
            raise IntegrationInitializationError("Failed to load tokenizer from model path.")

        self.model_id = settings["model_path"]

    def _prepare_prompt(self, messages: list[Message], tool_specs: Optional[list[dict[str, Any]]] = None) -> str:
        """
        Converts strongly typed Message objects to the dictionary format 
        expected by the MLX/HuggingFace tokenizer apply_chat_template.
        """
        msg_dicts: list[dict[str, str]] = []
        
        for m in messages:
            role_str = str(m.role)

            # Ensure we get text content
            content_str = _render_message_content(m)
            
            msg_dicts.append({"role": role_str, "content": content_str})

        if tool_specs:
            # Phi-family chat templates expect available tools to be embedded as a JSON
            # string on the system message, rendered as `<|tool|>...<|/tool|>`. Other chat
            # templates that don't recognize this convention will simply ignore the extra key.
            tools_json = json.dumps(tool_specs)
            system_msg = next((m for m in msg_dicts if m["role"] == "system"), None)
            if system_msg is not None:
                system_msg["tools"] = tools_json
            else:
                msg_dicts.insert(0, {"role": "system", "content": "You are a helpful assistant.", "tools": tools_json})

        if self.message_preprocessor:
            msg_dicts = self.message_preprocessor(msg_dicts)

        if self.mlx_tokenizer is not None and hasattr(self.mlx_tokenizer, "apply_chat_template"):
            return self.mlx_tokenizer.apply_chat_template(
                msg_dicts, 
                tokenize=False, 
                add_generation_prompt=True
            ) # type: ignore
        
        # Fallback
        return "\n".join([f"{m['role']}: {m['content']}" for m in msg_dicts])

    def _get_sampler(self, options: MLXChatOptions = {}):
        """Creates the MLX sampler, overriding defaults with ChatOptions if provided."""
        config = self.generation_config.model_dump()
        
        if options:
            if (temp := options.get("temperature")) is not None:
                config["temp"] = temp
            if (top_p := options.get("top_p")) is not None:
                config["top_p"] = top_p
            
            # Additional properties are now directly in the TypedDict
            if (min_p := options.get("min_p")) is not None:
                config["min_p"] = float(min_p)
            if (top_k := options.get("top_k")) is not None:
                config["top_k"] = int(top_k)
            if (xtc_prob := options.get("xtc_probability")) is not None:
                config["xtc_probability"] = float(xtc_prob)
            if (xtc_thresh := options.get("xtc_threshold")) is not None:
                config["xtc_threshold"] = float(xtc_thresh)

        return make_sampler(
            temp=config["temp"],
            top_p=config["top_p"],
            min_p=config["min_p"],
            min_tokens_to_keep=config["min_tokens_to_keep"],
            top_k=config["top_k"],
            xtc_probability=config["xtc_probability"],
            xtc_threshold=config["xtc_threshold"]
        )

    def _get_logits_processors(self, options: MLXChatOptions = {}):
        """Creates the MLX logits processors."""
        config = self.generation_config.model_dump()
        
        if options:
            if (rep_pen := options.get("repetition_penalty")) is not None:
                config["repetition_penalty"] = float(rep_pen)
            if (rep_ctx := options.get("repetition_context_size")) is not None:
                config["repetition_context_size"] = int(rep_ctx)

        return make_logits_processors(
            repetition_penalty=config.get("repetition_penalty"),
            repetition_context_size=config.get("repetition_context_size")
        )

    def _inner_get_response(
        self, 
        *, 
        messages: MutableSequence[Message],
        stream: bool = False,
        options: MLXChatOptions = {}, 
        **kwargs: Any
    ):
        if self.mlx_tokenizer is None:
             raise ValueError("Tokenizer is not initialized.")

        # Agent-level instructions arrive via options (per-provider handling is expected),
        # not as a Message, so turn them into a leading system message ourselves.
        prepped_messages = prepend_instructions_to_messages(list(messages), options.get("instructions"))

        tool_specs = self._build_tool_specs(options)
        prompt = self._prepare_prompt(prepped_messages, tool_specs)
        sampler = self._get_sampler(options)
        logits_processors = self._get_logits_processors(options)
        
        # Determine max_tokens: Option -> Config -> Default
        max_tokens = self.generation_config.max_tokens
        if (opt_max_tokens := options.get("max_tokens")) is not None:
             max_tokens = opt_max_tokens

        seed = self.generation_config.seed
        if (opt_seed := options.get("seed")) is not None:
            seed = int(opt_seed) 
        
        generate_kwargs = {}
        if seed is not None:
            generate_kwargs["seed"] = seed

        if stream:
            return ResponseStream(
                self._stream(prompt, max_tokens, sampler, logits_processors, generate_kwargs, tool_specs),
                finalizer=ChatResponse.from_updates
            )

        return self._do_get_response(prompt, max_tokens, sampler, logits_processors, generate_kwargs, options, tool_specs)

    def _build_tool_specs(self, options: MLXChatOptions) -> Optional[list[dict[str, Any]]]:
        """Converts the tools attached to this request into Phi-style function specs, if any."""
        tools = options.get("tools") if options else None
        if not tools:
            return None
        specs = [spec for tool in normalize_tools(tools) if (spec := _tool_to_function_spec(tool)) is not None]
        return specs or None

    async def _do_get_response(
        self,
        prompt: str,
        max_tokens: int,
        sampler: Any,
        logits_processors: Any,
        generate_kwargs: dict,
        options: MLXChatOptions,
        tool_specs: Optional[list[dict[str, Any]]] = None,
    ):
        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(
            self._executor,
            functools.partial(
                generate,
                model=self.model,
                tokenizer=self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=self.generation_config.verbose,
                **generate_kwargs
            )
        )

        # 1. Parse the raw completion into Content items: a list of function_call contents
        #    if the model produced a tool call (only attempted when tools were offered),
        #    otherwise a single text content.
        contents = self._parse_response_contents(response_text, tool_specs)
        
        # 2. Create Message
        message = Message(
            "assistant",
            contents
        )

        # 3. Calculate usage
        prompt_tokens = len(self.mlx_tokenizer.encode(prompt)) # type: ignore
        completion_tokens = len(self.mlx_tokenizer.encode(response_text)) # type: ignore
        usage = UsageDetails(
            input_token_count=prompt_tokens,
            output_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens
        )

        # 4. Create ChatResponse
        return ChatResponse(
            messages=[message],
            model=self.model_id,
            usage_details=usage
        )

    def _parse_response_contents(
        self,
        response_text: str,
        tool_specs: Optional[list[dict[str, Any]]],
    ) -> list[Content]:
        """Turns a raw completion into function_call contents (if a tool call was made
        and tools were offered for this request) or a single text content otherwise."""
        if tool_specs:
            tool_calls = _extract_tool_calls(response_text)
            if tool_calls:
                return [
                    Content.from_function_call(
                        call_id=str(uuid.uuid4()),
                        name=call["name"],
                        arguments=call.get("arguments"),
                    )
                    for call in tool_calls
                ]
        return [Content.from_text(text=response_text)]

    async def _stream(
        self,
        prompt: str,
        max_tokens: int,
        sampler: Any,
        logits_processors: Any,
        generate_kwargs: dict,
        tool_specs: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterable[ChatResponseUpdate]:
        if self.mlx_tokenizer is None:
             raise ValueError("Tokenizer is not initialized.")

        # Get the synchronous generator from MLX
        # We need to run this in a thread to avoid blocking the event loop
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                generation_stream = stream_generate(
                    model=self.model,
                    tokenizer=self.mlx_tokenizer, # type: ignore
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    **generate_kwargs
                )
                for response_chunk in generation_stream:
                    loop.call_soon_threadsafe(queue.put_nowait, response_chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)

        # Start the producer on the dedicated MLX worker thread (matches the thread the
        # model was loaded on, avoiding cross-thread MLX stream errors).
        self._executor.submit(producer)

        # Consume from the queue
        last_usage = None
        full_text = ""
        # When tools are offered, the model may respond with raw tool-call JSON instead of
        # prose. Buffer the chunks and decide how to surface them only once generation has
        # finished, so callers never see partial/raw tool-call syntax mid-stream.
        buffer_for_tools = bool(tool_specs)
        pending_updates: list[ChatResponseUpdate] = []

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            
            # Extract usage if available
            if hasattr(item, "prompt_tokens") and hasattr(item, "generation_tokens"):
                last_usage = UsageDetails(
                    input_token_count=item.prompt_tokens,
                    output_token_count=item.generation_tokens,
                    total_token_count=item.prompt_tokens + item.generation_tokens
                )

            full_text += item.text
            update = ChatResponseUpdate(
                role="assistant", 
                contents=[Content.from_text(text=item.text)], 
                model=self.model_id
            )
            if buffer_for_tools:
                pending_updates.append(update)
            else:
                yield update

        if buffer_for_tools:
            tool_calls = _extract_tool_calls(full_text)
            if tool_calls:
                for call in tool_calls:
                    yield ChatResponseUpdate(
                        role="assistant",
                        contents=[Content.from_function_call(
                            call_id=str(uuid.uuid4()),
                            name=call["name"],
                            arguments=call.get("arguments"),
                        )],
                        model=self.model_id
                    )
            else:
                for update in pending_updates:
                    yield update

        # Yield usage at the end if we captured it
        if last_usage:
            yield ChatResponseUpdate(
                role="assistant",
                contents=[Content.from_usage(usage_details=last_usage)],
                model=self.model_id
            )