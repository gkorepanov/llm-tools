from typing import AsyncIterator, Dict, List

import openai


async def make_generator(
    messages: List[Dict[str, str]],
    stream: bool = True,
    **kwargs
) -> AsyncIterator[str]:
    kwargs['temperature'] = kwargs.get('temperature', 0)
    kwargs['model'] = kwargs.get('model', "gpt-3.5-turbo")
    if stream:
        async for chunk in await openai.ChatCompletion.acreate(
            stream=True,
            messages=messages,
            **kwargs,
        ):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                yield content
    else:
        result = await openai.ChatCompletion.acreate(
            stream=False,
            messages=messages,
            **kwargs,
        )
        yield result["choices"][0].message["content"]
