from typing import (
    AsyncIterator,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    retry_if_exception,
)
import asyncio
from tenacity.wait import wait_base
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import BaseMessage
import tiktoken


from llm_tools.errors import (
    should_retry_initital_openai_request_error,
    should_retry_streaming_openai_request_error,
    should_fallback_to_other_model,
    get_openai_retrying_iterator,
    ModelContextSizeExceededError,
    StreamingNextTokenTimeoutError,
)


def reset_openai_globals():
    import openai
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = None
    openai.api_version = None


@dataclass
class TokenExpense:
    model_name: str
    n_input_tokens: int = 0
    n_output_tokens: int = 0

    @property
    def n_total_tokens(self) -> int:
        return self.n_input_tokens + self.n_output_tokens
    
    def __add__(self, other: "TokenExpense") -> "TokenExpense":
        if other.model_name != self.model_name:
            raise ValueError("Cannot add TokenExpense objects with different model names")
        return TokenExpense(
            n_input_tokens=self.n_input_tokens + other.n_input_tokens,
            n_output_tokens=self.n_output_tokens + other.n_output_tokens,
        )


class StreamingOpenAIChatModel:
    def __init__(
        self,
        chat_model: Union[ChatOpenAI, AzureChatOpenAI],
        max_initial_request_retries: int = 5,
        max_streaming_retries: int = 2,
        wait_between_retries=wait_exponential(multiplier=1, min=1, max=60),
        streaming_next_token_timeout: int = 10,
        request_timeout: wait_base = wait_exponential(multiplier=1, min=5, max=60),
    ):
        self.chat_model = chat_model
        self.encoding = tiktoken.encoding_for_model(self.chat_model.model_name)
        self.max_request_retries = max_initial_request_retries
        self.max_streaming_retries = max_streaming_retries
        self.wait_between_retries = wait_between_retries
        self.streaming_next_token_timeout = streaming_next_token_timeout
        self.request_timeout = request_timeout
        self.reset()

    def reset(self):
        self.completions = []
        self.successful_request_attempts = 0
        self.request_attempts = 0
        self.streaming_attempts = 0
        self.message_dicts = None
        self.succeeded = False

    async def stream_llm_reply(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        assert self.chat_model.streaming
        self.reset()
        self.message_dicts, params = self.chat_model._create_message_dicts(messages, stop)
        params["stream"] = True

        async for streaming_attempt in get_openai_retrying_iterator(
            retry_if_exception_fn=should_retry_streaming_openai_request_error,
            max_retries=self.max_streaming_retries,
            wait=self.wait_between_retries,
        ):
            completion = ""
            role = "assistant"
            self.streaming_attempts += 1

            async for request_attempt in get_openai_retrying_iterator(
                retry_if_exception_fn=should_retry_initital_openai_request_error,
                max_retries=self.max_request_retries,
                wait=self.wait_between_retries,
            ):
                with request_attempt:
                    self.request_attempts += 1
                    timeout = self.request_timeout(request_attempt.retry_state)
                    try:
                        gen = await asyncio.wait_for(
                            self.chat_model.client.acreate(messages=self.message_dicts, **params),
                            timeout=timeout,
                        )
                    except:
                        raise
                    else:
                        self.successful_request_attempts += 1

            with streaming_attempt:
                try:
                    gen_iter = gen.__aiter__()
                    while True:
                        try:
                            stream_resp = await asyncio.wait_for(
                                gen_iter.__anext__(),
                                timeout=self.streaming_next_token_timeout,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            raise StreamingNextTokenTimeoutError()
                        finish_reason = stream_resp["choices"][0].get("finish_reason")
                        role = stream_resp["choices"][0]["delta"].get("role", role)
                        token = stream_resp["choices"][0]["delta"].get("content", "")
                        completion += token
                        if token:
                            yield completion, token
                        if finish_reason and finish_reason != "stop":
                            raise ModelContextSizeExceededError()
                finally:
                    self.completions.append(completion)

        self.succeeded = True
    
    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> TokenExpense:
        if not self.succeeded and only_successful_trial:
            raise ValueError("Cannot get tokens spent for unsuccessful trial")

        n_input_tokens_per_trial = self._count_tokens_from_input_messages(self.message_dicts)
        if only_successful_trial:
            n_input_tokens = n_input_tokens_per_trial
            n_output_tokens = self._count_tokens_from_output_text(self.completions[-1])
        else:
            n_input_tokens = n_input_tokens_per_trial * self.successful_request_attempts
            n_output_tokens = sum(self._count_tokens_from_output_text(completion) for completion in self.completions)
        return TokenExpense(
            n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens,
            model_name=self.chat_model.model_name,
        )

    def _count_tokens_from_input_messages(
        self,
        messages: List[Any],
    ) -> int:
        if not messages:
            return 0
        model = self.chat_model.model_name

        if model == "gpt-3.5-turbo":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(self.encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2
        return n_input_tokens
    
    def _count_tokens_from_output_text(
        self,
        text: str,
    ) -> int:
        if not text:
            return 0

        return 1 + len(self.encoding.encode(text))
    


class MultipleException(Exception):
    def __init__(
        self,
        exceptions: List[Exception],
    ):
        self.exceptions = exceptions
    
    def __str__(self):
        return "\n".join(
            f"{type(e).__name__}: {str(e)}"
            for e in self.exceptions
        )



class StreamingModelWithFallback:
    def __init__(
        self,
        models: List[StreamingOpenAIChatModel],
        should_fallback_to_other_model: Callable[[Exception], bool] = should_fallback_to_other_model, 
    ):
        self.models = models
        self.should_fallback_to_other_model = should_fallback_to_other_model
        self.exceptions = []
    
    async def stream_llm_reply(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        self.exceptions = []
        for model in self.models:
            try:
                async for completion, token in model.stream_llm_reply(messages, stop):
                    yield completion, token
            except Exception as e:
                if self.should_fallback_to_other_model(e):
                    self.exceptions.append(e)
                    continue
                else:
                    raise
            else:
                break
        else:
            if len(self.exceptions) == 1:
                raise self.exceptions[0]
            else:
                raise MultipleException(self.exceptions)

    @property
    def succeeded(self) -> bool:
        return any(model.succeeded for model in self.models)

    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> List[TokenExpense]:
        
        if not self.succeeded and only_successful_trial:
            raise ValueError("Cannot get tokens spent for unsuccessful trial")
            
        if only_successful_trial:
            first_successful_model = next(model for model in self.models if model.succeeded)
            return [first_successful_model.get_tokens_spent(only_successful_trial)]
        else:
            return [
                model.get_tokens_spent(only_successful_trial)
                for model in self.models
            ]
