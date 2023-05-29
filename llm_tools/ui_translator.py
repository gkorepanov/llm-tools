from typing import List, Dict, Iterable, AsyncIterator
from iso639 import Lang
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed

from llm_tools.generator import make_generator


DEFAULT_LANGUAGES = tuple(
    Lang(x) for x in ("en", "ru", "es", "id", "pt", "de", "fr")
)


system_task = "You are a professional translator who helps to translate UI texts"
user_task_template = """I need you to translate the UI text into following languages: {languages}. Respond with a yaml formatted like
{yaml_example}.
Use multiline yaml format if needed. The English UI text you need to translate is:
\"\"\"{text}\"\"\"{extra_prompt}"""


@retry(stop=stop_after_attempt(4), wait=wait_fixed(1))
async def get_translation_yaml(
    text: str,
    languages: List[Lang],
    extra_prompt: str = "",
    stream: bool = True,
) -> AsyncIterator[str]:
    yaml_example = []
    for x in languages:
        yaml_example.append(f"{x.pt1}: |\n  <text>")

    yaml_example = '\n'.join(yaml_example)

    user_task = user_task_template.format(
        text=text,
        yaml_example=yaml_example,
        languages=', '.join(x.name for x in languages),
        extra_prompt=extra_prompt,
    )

    messages = [
        {"role": "system", "content": system_task},
        {"role": "user", "content": user_task},
    ]
    line = []
    async for token in make_generator(messages, model='gpt-4', stream=stream):
        yield token


async def get_full_translation_yaml(
    english_strings: Dict[str, str],
    languages: Iterable[Lang] = DEFAULT_LANGUAGES,
    extra_prompt: str = "",
    stream: bool = True,
) -> AsyncIterator[str]:
    languages = list(languages)
    for name, text in english_strings.items():
        yield f"{name}:\n  "
        async for token in get_translation_yaml(
            text,
            languages,
            extra_prompt=extra_prompt,
            stream=stream,
        ):
            lines = token.split('\n')
            yield lines[0]
            for line in lines[1:]:
                yield '\n  ' + line
        yield '\n'


async def get_full_translation_yaml_parallel(
    english_strings: Dict[str, str],
    languages: Iterable[Lang] = DEFAULT_LANGUAGES,
    extra_prompt: str = "",
    stream: bool = True,
) -> AsyncIterator[str]:
    languages = list(languages)
    async def _get_result(name: str, text: str) -> str:
        result = ""
        result += f"{name}:\n  "
        async for token in get_translation_yaml(
            text,
            languages,
            extra_prompt=extra_prompt,
            stream=stream,
        ):
            lines = token.split('\n')
            result += lines[0]
            for line in lines[1:]:
                result += '\n  ' + line
        return result

    # yield results as they come
    tasks = []
    for name, text in english_strings.items():
        tasks.append(_get_result(name, text))
    for task in asyncio.as_completed(tasks):
        yield await task
