from typing import (
    AsyncIterator,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from tenacity import AsyncRetrying
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
    BaseLanguageModel
)
import tiktoken


from llm_tools.errors import (
    OPENAI_STREAMING_RETRING_ITERATOR_FN,
    OPENAI_REQUEST_RETRING_ITERATOR_FN,
)


@dataclass
class TokenExpense:
    n_input_tokens: int
    n_output_tokens: int

    @property
    def n_total_tokens(self) -> int:
        return self.n_input_tokens + self.n_output_tokens
    
    def __add__(self, other: "TokenExpense") -> "TokenExpense":
        return TokenExpense(
            n_input_tokens=self.n_input_tokens + other.n_input_tokens,
            n_output_tokens=self.n_output_tokens + other.n_output_tokens,
        )


class StreamingOpenAIChatModel:
    def __init__(
        self,
        chat_model: Union[ChatOpenAI, AzureChatOpenAI],
        streaming_retrying_iterator_fn: Callable[[], AsyncRetrying] = OPENAI_STREAMING_RETRING_ITERATOR_FN,
        request_retrying_iterator_fn: Callable[[], AsyncRetrying] = OPENAI_REQUEST_RETRING_ITERATOR_FN,
    ):
        self.chat_model = chat_model
        self.streaming_retrying_iterator_fn = streaming_retrying_iterator_fn
        self.request_retrying_iterator_fn = request_retrying_iterator_fn
        self.encoding = tiktoken.encoding_for_model(self.chat_model.model_name)
        self.reset()
    
    def reset(self):
        self.completions = []
        self.request_attempts = 0
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

        async for streaming_attempt in self.streaming_retrying_iterator_fn():
            completion = ""
            role = "assistant"
            async for request_attempt in self.request_retrying_iterator_fn():
                with request_attempt:
                    self.request_attempts += 1
                    gen = await self.chat_model.client.acreate(messages=self.message_dicts, **params)

            with streaming_attempt:
                try:
                    async for stream_resp in gen:
                        role = stream_resp["choices"][0]["delta"].get("role", role)
                        token = stream_resp["choices"][0]["delta"].get("content", "")
                        completion += token
                        if token:
                            yield completion, token
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
            n_input_tokens = sum(n_input_tokens_per_trial for _ in self.completions)
            n_output_tokens = sum(self._count_tokens_from_output_text(completion) for completion in self.completions)
        return TokenExpense(
            n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens,
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
        return "\n".join(str(e) for e in self.exceptions)



class StreamingModelWithFallback:
    def __init__(
        self,
        models: List[StreamingOpenAIChatModel],
    ):
        self.models = models
        self.succe
    
    async def stream_llm_reply(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        exceptions = []
        for model in self.models:
            try:
                async for completion, token in model.stream_llm_reply(messages, stop):
                    yield completion, token
            except Exception as e:
                exceptions.append(e)
            else:
                break
        else:
            raise MultipleException(exceptions)
        
    @property
    def succeeded(self) -> bool:
        return any(model.succeeded for model in self.models)

    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> TokenExpense:
        
        if not self.succeeded and only_successful_trial:
            raise ValueError("Cannot get tokens spent for unsuccessful trial")
            
        if only_successful_trial:
            first_successful_model = next(model for model in self.models if model.succeeded)
            return first_successful_model.get_tokens_spent(only_successful_trial)
        else:
            return sum(model.get_tokens_spent(only_successful_trial) for model in self.models)
