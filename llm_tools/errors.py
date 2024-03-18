from tenacity import (
    before_sleep_log,
    stop_after_attempt,
    AsyncRetrying,
    retry_if_exception,
)
from tenacity.wait import wait_base
from typing import Callable, Optional, List
import logging
import openai
import aiohttp
import aiohttp.client_exceptions
import asyncio
import re

from litellm import ContextWindowExceededError


logger = logging.getLogger(__name__)



class ModelContextSizeExceededError(Exception):
    def __init__(
        self,
        model_name: str,
        max_context_length: int,
        context_length: Optional[int] = None,
        during_streaming: bool = False,
    ):
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.context_length = context_length
        self.during_streaming = during_streaming

    def __str__(self) -> str:
        suffix = " (during streaming)" if self.during_streaming else ""
        if self.context_length is None:
            return f"Context length exceeded for model {self.model_name}{suffix}"
        else:
            return f"Context length exceeded for model {self.model_name}{suffix}: {self.context_length} > {self.max_context_length}"

    @classmethod
    def from_litellm_error(
        cls,
        error: ContextWindowExceededError,
        model_name: str,
        during_streaming: bool = False,
    ) -> "ModelContextSizeExceededError":
        max_context_length_pattern = r"maximum context length is (\d+) tokens"
        tokens_number_pattern = r"messages resulted in (\d+) tokens"

        max_context_length = re.search(max_context_length_pattern, str(error))
        tokens_number = re.search(tokens_number_pattern, str(error))

        if max_context_length is not None:
            max_context_length = int(max_context_length.group(1))

        if tokens_number is not None:
            tokens_number = int(tokens_number.group(1))

        return ModelContextSizeExceededError(
            model_name=model_name,
            max_context_length=max_context_length,
            context_length=tokens_number,
            during_streaming=during_streaming,
        )


class StreamingNextTokenTimeoutError(asyncio.TimeoutError):
    pass


class OpenAIRequestTimeoutError(asyncio.TimeoutError):
    pass


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


CONTEXT_LENGTH_EXCEEDED_ERROR_CODE = "context_length_exceeded"


def should_retry_initital_openai_request_error(error: Exception) -> bool:
    OPENAI_REQUEST_ERRORS = (
        openai.APIError,
    )
    return isinstance(error, OPENAI_REQUEST_ERRORS)


def should_retry_streaming_openai_request_error(error: Exception) -> bool:
    OPENAI_STREAMING_ERRORS = (
        openai.APIError,
        aiohttp.client_exceptions.ClientPayloadError,
        StreamingNextTokenTimeoutError,
    )
    return isinstance(error, OPENAI_STREAMING_ERRORS)


def should_fallback_to_other_model(error: Exception) -> bool:
    if isinstance(error, ModelContextSizeExceededError):
        return False

    if isinstance(error, ContextWindowExceededError):
        return False

    return True


def get_openai_retrying_iterator(
    retry_if_exception_fn: Callable[[Exception], bool],
    wait: wait_base,
    max_retries: int = 1,
) -> AsyncRetrying:
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait,
        retry=retry_if_exception(retry_if_exception_fn),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
