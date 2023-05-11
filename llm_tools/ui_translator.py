from typing import List, Dict, Iterable, AsyncIterator
from iso639 import Lang

from llm_tools.generator import make_generator


DEFAULT_LANGUAGES = tuple(
    Lang(x) for x in ("en", "ru", "es", "id", "pt", "de", "fr")
)


system_task = "You are a professional translator who helps to translate UI texts"
user_task_template = """I need you to translate the UI text into following languages: {languages}. Respond with a yaml formatted like
{yaml_example}.
Use multiline yaml format if needed. The English UI text you need to translate is:
\"\"\"{text}\"\"\""""


async def get_translation_yaml(text: str, languages: List[Lang]) -> AsyncIterator[str]:
    yaml_example = []
    for x in languages:
        yaml_example.append(f"{x.pt1}: |\n  <text>")

    yaml_example = '\n'.join(yaml_example)

    user_task = user_task_template.format(
        text=text,
        yaml_example=yaml_example,
        languages=', '.join(x.name for x in languages)
    )

    messages = [
        {"role": "system", "content": system_task},
        {"role": "user", "content": user_task},
    ]
    line = []
    async for token in make_generator(messages, model='gpt-4'):
        yield token


async def get_full_translation_yaml(
    english_strings: Dict[str, str],
    languages: Iterable[Lang] = DEFAULT_LANGUAGES,
) -> AsyncIterator[str]:
    languages = list(languages)
    for name, text in english_strings.items():
        yield f"{name}:\n  "
        async for token in get_translation_yaml(text, languages):
            lines = token.split('\n')
            yield lines[0]
            for line in lines[1:]:
                yield '\n  ' + line
        yield '\n'
