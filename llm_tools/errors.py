from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
)
from typing import Any, Callable, Tuple
import logging
import openai
import openai.error
import aiohttp
import aiohttp.client_exceptions


logger = logging.getLogger(__name__)


OPENAI_REQUEST_ERRORS = (
    openai.error.Timeout,
    openai.error.APIError,
    openai.error.APIConnectionError,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
)

OPENAI_STREAMING_ERRORS = (
    aiohttp.client_exceptions.ClientPayloadError,
)


def get_openai_retrying_iterator(
    errors: Tuple[Exception, ...],
    max_retries: int = 5,
    min_seconds: int = 1,
    max_seconds: int = 60,
) -> AsyncRetrying:
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_if_exception_type(OPENAI_REQUEST_ERRORS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

OPENAI_REQUEST_RETRING_ITERATOR_FN = lambda: get_openai_retrying_iterator(OPENAI_REQUEST_ERRORS)
OPENAI_STREAMING_RETRING_ITERATOR_FN = lambda: get_openai_retrying_iterator(OPENAI_STREAMING_ERRORS)
