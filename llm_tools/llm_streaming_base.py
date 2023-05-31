from typing import (
    AsyncIterator,
    List,
    Optional,
    Tuple,
)

from llm_tools.tokens import TokenExpense
from llm_tools.chat_message import OpenAIChatMessage


class StreamingLLMBase(object):
    async def stream_llm_reply(
        self,
        messages: List[OpenAIChatMessage],
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        raise NotImplementedError()
    
    @property
    def succeeded(self) -> bool:
        raise NotImplementedError()
    
    def get_tokens_spent(
        self,
        only_successful_trial: bool = False,
    ) -> List[TokenExpense]:
        raise NotImplementedError()
