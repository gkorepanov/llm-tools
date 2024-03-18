from typing import (
    AsyncIterator,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from tenacity import wait_exponential
import asyncio
from tenacity.wait import wait_base

import openai
from litellm import get_max_tokens, get_model_info, acompletion, token_counter, ContextWindowExceededError

from concurrent.futures import Executor
from functools import partial

from llm_tools.chat_message import OpenAIChatMessage, prepare_messages, convert_message_to_dict
from llm_tools.tokens import (
    TokenExpense,
    TokenExpenses,
)

from llm_tools.errors import (
    should_retry_initital_openai_request_error,
    should_retry_streaming_openai_request_error,
    get_openai_retrying_iterator,
    ModelContextSizeExceededError,
    StreamingNextTokenTimeoutError,
)
from llm_tools.llm_streaming_base import StreamingLLMBase



class StreamingChatModel(StreamingLLMBase):
    def __init__(
        self,
        model: str,
        temperature: float,
        api_key: str,
        max_tokens: Optional[int] = None,
        lite_llm_params: Optional[Dict[str, Any]] = None,
        max_initial_request_retries: int = 5,
        max_streaming_retries: int = 2,
        wait_between_retries=wait_exponential(multiplier=1, min=1, max=60),
        streaming_next_token_timeout: int = 10,
        request_timeout: wait_base = wait_exponential(multiplier=1, min=5, max=60),
        token_count_executor: Optional[Executor] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        if max_tokens is None:
            max_tokens = get_max_tokens(model)
        self.max_tokens_to_output = max_tokens
        self.lite_llm_params = lite_llm_params if lite_llm_params is not None else {}
        self.max_request_retries = max_initial_request_retries
        self.max_streaming_retries = max_streaming_retries
        self.wait_between_retries = wait_between_retries
        self.streaming_next_token_timeout = streaming_next_token_timeout
        self.request_timeout = request_timeout
        self.token_count_executor = token_count_executor
        self.reset()

    @property
    def max_tokens(self) -> int:
        return get_max_tokens(self.model)

    @property
    def context_size(self) -> int:
        i = get_model_info(self.model)
        res = i.get("max_input_tokens", i.get("max_tokens"))
        if res is None:
            raise ValueError(f"Could not get context size for model {self.model}")
        return res

    def reset(self):
        self.completions = []
        self.successful_request_attempts = 0
        self.request_attempts = 0
        self.streaming_attempts = 0
        self.message_dicts = None
        self._succeeded = False
        self.input_messages_n_tokens = 0
        self.output_tokens_spent_per_completion = []

    @property
    def succeeded(self) -> bool:
        return self._succeeded

    async def stream_llm_reply(
        self,
        messages: List[OpenAIChatMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        self.reset()

        assert len(messages) > 0
        messages = [
            convert_message_to_dict(x)
            for x in prepare_messages(messages)
        ]

        _f = partial(token_counter,
            messages=messages,
            model=self.model,
        )
        if self.token_count_executor is None:
            self.input_messages_n_tokens = _f()
        else:
            self.input_messages_n_tokens = await asyncio.get_running_loop().run_in_executor(
                self.token_count_executor,
                _f,
            )
        if self.input_messages_n_tokens > self.context_size:
            raise ModelContextSizeExceededError(
                model_name=self.chat_model.model_name,
                max_context_length=self.context_size,
                context_length=self.input_messages_n_tokens,
                during_streaming=False,
            )

        self.message_dicts = messages

        async for streaming_attempt in get_openai_retrying_iterator(
            retry_if_exception_fn=should_retry_streaming_openai_request_error,
            max_retries=self.max_streaming_retries,
            wait=self.wait_between_retries,
        ):
            completion = ""
            role = "assistant"
            self.streaming_attempts += 1
            self.output_tokens_spent_per_completion.append(0)

            async for request_attempt in get_openai_retrying_iterator(
                retry_if_exception_fn=should_retry_initital_openai_request_error,
                max_retries=self.max_request_retries,
                wait=self.wait_between_retries,
            ):
                with request_attempt:
                    self.request_attempts += 1
                    timeout = self.request_timeout(request_attempt.retry_state)
                    try:
                        gen = await acompletion(
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens_to_output,
                            messages=self.message_dicts,
                            **self.lite_llm_params,
                            timeout=timeout,
                            stop=stop,
                            api_key=self.api_key,
                            stream=True,
                        )
                    except ContextWindowExceededError as e:
                        raise ModelContextSizeExceededError.from_litellm_error(
                            model_name=self.model,
                            during_streaming=False,
                            error=e,
                        ) from e
                    except asyncio.TimeoutError as e:  # map exceptions to OpenAIExceptions
                        raise openai.APITimeoutError from e
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
                        except asyncio.TimeoutError as e:
                            raise StreamingNextTokenTimeoutError() from e
                        finish_reason = stream_resp["choices"][0].get("finish_reason")
                        role = stream_resp["choices"][0]["delta"].get("role", role)
                        token = stream_resp["choices"][0]["delta"].get("content", "") or ""

                        _f = partial(token_counter,
                            text=token,
                            model=self.model,
                            count_response_tokens=True,
                        )
                        if self.token_count_executor is None:
                            _tokens = _f()
                        else:
                            _tokens = await asyncio.get_running_loop().run_in_executor(
                                self.token_count_executor,
                                _f,
                            )
                        self.output_tokens_spent_per_completion[-1] += _tokens
                        completion += token
                        if token:
                            yield completion, token
                        if finish_reason:
                            if finish_reason == "length":
                                raise ModelContextSizeExceededError(
                                    model_name=self.model,
                                    max_context_length=self.context_size,
                                    context_length=self.input_messages_n_tokens + self.output_tokens_spent_per_completion[-1],
                                    during_streaming=True,
                                )
                            elif finish_reason != "stop":
                                raise ValueError(f"Unknown finish reason: {finish_reason}")
                finally:
                    self.completions.append(completion)

        self._succeeded = True

    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> TokenExpenses:
        if not self.succeeded and only_successful_trial:
            raise ValueError("Cannot get tokens spent for unsuccessful trial")

        n_input_tokens_per_trial = self.input_messages_n_tokens
        if only_successful_trial:
            n_input_tokens = n_input_tokens_per_trial
            n_output_tokens = self.output_tokens_spent_per_completion[-1]
        else:
            n_input_tokens = n_input_tokens_per_trial * self.successful_request_attempts
            n_output_tokens = sum(self.output_tokens_spent_per_completion)
        expenses = TokenExpenses()
        expense = TokenExpense(
            n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens,
            model_name=self.model,
        )
        expenses.add_expense(expense)
        return expenses
