import asyncio
import logging
from typing import Any, AsyncIterable, MutableSequence, Optional, Callable, ClassVar, TypedDict
from pydantic import BaseModel
from agent_framework import (
    BaseChatClient, 
    ChatMessage, 
    ChatOptions, 
    ChatResponse, 
    ChatResponseUpdate, 
    Role, 
    Content,
    UsageDetails,
    use_chat_middleware,
    use_function_invocation
)
from agent_framework.observability import use_instrumentation
from agent_framework._pydantic import AFBaseSettings
from agent_framework.exceptions import ServiceInitializationError
from mlx_lm.utils import load
from mlx_lm.generate import generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
import threading

logger = logging.getLogger(__name__)

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

class MLXSettings(AFBaseSettings):
    """
    MLX Client settings.
    
    Attributes:
        model_path: The path to the MLX model. (Env var MLX_MODEL_PATH)
        adapter_path: Optional path to an adapter. (Env var MLX_ADAPTER_PATH)
    """
    env_prefix: ClassVar[str] = "MLX_"
    
    model_path: str
    adapter_path: Optional[str] = None

class MLXChatOptions(ChatOptions, total=False):
    """MLX-specific Chat Options."""
    min_p: float
    top_k: int
    xtc_probability: float
    xtc_threshold: float
    repetition_penalty: float
    repetition_context_size: int

@use_function_invocation
@use_instrumentation
@use_chat_middleware
class MLXChatClient(BaseChatClient[MLXChatOptions]):
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
        settings = MLXSettings(
            model_path=model_path, # type: ignore
            adapter_path=adapter_path,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding
        )

        super().__init__(**kwargs)
        
        self.generation_config = generation_config or MLXGenerationConfig()
        self.message_preprocessor = message_preprocessor
        
        logger.info(f"Loading MLX Model: {settings.model_path}...")
        loaded = load(
            settings.model_path,
            adapter_path=settings.adapter_path,
            tokenizer_config=tokenizer_config or {}
        ) # type: ignore
        
        # Handle variable return length from mlx_lm.load
        if isinstance(loaded, tuple):
            self.model = loaded[0]
            self.tokenizer = loaded[1]
        else:
            self.model = loaded
            self.tokenizer = None 

        if self.tokenizer is None:
            raise ServiceInitializationError("Failed to load tokenizer from model path.")

        self.model_id = settings.model_path

    def _prepare_prompt(self, messages: list[ChatMessage]) -> str:
        """
        Converts strongly typed ChatMessage objects to the dictionary format 
        expected by the MLX/HuggingFace tokenizer apply_chat_template.
        """
        msg_dicts: list[dict[str, str]] = []
        
        for m in messages:
            role_str: str
            if isinstance(m.role, Role):
                role_str = m.role.value
            elif isinstance(m.role, str):
                role_str = m.role
            else:
                # Fallback for dict representation or other EnumLikes
                role_str = str(m.role)

            # Ensure we get text content
            content_str = m.text if hasattr(m, "text") else str(m.contents)
            
            msg_dicts.append({"role": role_str, "content": content_str})

        if self.message_preprocessor:
            msg_dicts = self.message_preprocessor(msg_dicts)

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
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

    async def _inner_get_response(
        self, 
        *, 
        messages: MutableSequence[ChatMessage], 
        options: MLXChatOptions = {}, 
        **kwargs: Any
    ) -> ChatResponse:
        
        if self.tokenizer is None:
             raise ValueError("Tokenizer is not initialized.")

        prompt = self._prepare_prompt(list(messages))
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

        response_text = await asyncio.to_thread(
            generate,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=self.generation_config.verbose,
            **generate_kwargs
        )

        # 1. Create TextContent
        content = Content.from_text(text=response_text)
        
        # 2. Create ChatMessage
        message = ChatMessage(
            role=Role.ASSISTANT, 
            contents=[content]
        )

        # 3. Calculate usage
        prompt_tokens = len(self.tokenizer.encode(prompt)) # type: ignore
        completion_tokens = len(self.tokenizer.encode(response_text)) # type: ignore
        usage = UsageDetails(
            input_token_count=prompt_tokens,
            output_token_count=completion_tokens,
            total_token_count=prompt_tokens + completion_tokens
        )

        # 4. Create ChatResponse
        return ChatResponse(
            messages=[message],
            model_id=self.model_id,
            usage_details=usage
        )

    async def _inner_get_streaming_response(
        self, 
        *, 
        messages: MutableSequence[ChatMessage],  
        options: MLXChatOptions = {}, 
        **kwargs: Any
    ) -> AsyncIterable[ChatResponseUpdate]:
        
        if self.tokenizer is None:
             raise ValueError("Tokenizer is not initialized.")

        prompt = self._prepare_prompt(list(messages))
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

        # Get the synchronous generator from MLX
        # We need to run this in a thread to avoid blocking the event loop
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                generation_stream = stream_generate(
                    model=self.model,
                    tokenizer=self.tokenizer, # type: ignore
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

        # Start the producer thread
        thread = threading.Thread(target=producer)
        thread.start()

        # Consume from the queue
        last_usage = None

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

            content = Content.from_text(text=item.text)
            
            yield ChatResponseUpdate(
                role=Role.ASSISTANT, 
                contents=[content], 
                model_id=self.model_id
            )
        
        # Yield usage at the end if we captured it
        if last_usage:
            yield ChatResponseUpdate(
                role=Role.ASSISTANT,
                contents=[Content.from_usage(usage_details=last_usage)],
                model_id=self.model_id
            )