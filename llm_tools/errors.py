from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    retry_if_exception,
)
from tenacity.wait import wait_base
from typing import Any, Callable, Tuple
import logging
import openai
import openai.error
import aiohttp
import aiohttp.client_exceptions
import asyncio


logger = logging.getLogger(__name__)



class ModelContextSizeExceededError(Exception):
    pass


class StreamingNextTokenTimeoutError(asyncio.TimeoutError):
    pass


class OpenAIRequestTimeoutError(asyncio.TimeoutError):
    pass


CONTEXT_LENGTH_EXCEEDED_ERROR_CODE = "context_length_exceeded"


def should_retry_initital_openai_request_error(error: Exception) -> bool:
    OPENAI_REQUEST_ERRORS = (
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        OpenAIRequestTimeoutError,
    )
    return isinstance(error, OPENAI_REQUEST_ERRORS)


def should_retry_streaming_openai_request_error(error: Exception) -> bool:
    OPENAI_STREAMING_ERRORS = (
        aiohttp.client_exceptions.ClientPayloadError,
        StreamingNextTokenTimeoutError,
    )
    return isinstance(error, OPENAI_STREAMING_ERRORS)


def should_fallback_to_other_model(error: Exception) -> bool:
    if isinstance(error, ModelContextSizeExceededError):
        return False
    
    if isinstance(error, openai.error.InvalidRequestError) and error.code == CONTEXT_LENGTH_EXCEEDED_ERROR_CODE:
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
