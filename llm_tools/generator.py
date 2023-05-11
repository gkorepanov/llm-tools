from typing import AsyncIterator, Dict, List

import openai


async def make_generator(messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
    kwargs['temperature'] = kwargs.get('temperature', 0)
    kwargs['model'] = kwargs.get('model', "gpt-3.5-turbo")
    async for chunk in await openai.ChatCompletion.acreate(
        stream=True,
        messages=messages,
        **kwargs,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content
