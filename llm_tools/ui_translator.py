from typing import List, Dict
from iso639 import Lang



DEFAULT_LANGUAGES = tuple(
    Lang(x) for x in ("en", "ru", "es", "id", "pt", "de", "fr")
)


system_task = "You are a professional translator who helps to translate UI texts"
user_task_template = """I need you to translate the UI text into following languages: {languages}. Respond with a yaml formatted in the following way:
```yaml
{yaml_example}
```
The UI text you need to translate is:
```yaml
{example}
```
Only respond with a raw yaml, omit any comments. Include only this languages into output: {language_codes}
"""



def get_translation_yaml(
    text: str,
    languages: List[Lang],
) -> List[Dict[str, str]]:
    yaml_example = []
    for x in languages:
        yaml_example.append(f"{x.pt1}: |-\n  <text>")

    yaml_example = '\n'.join(yaml_example)

    user_task = user_task_template.format(
        example=text,
        yaml_example=yaml_example,
        languages=', '.join(x.name for x in languages),
        language_codes=', '.join(x.pt1 for x in languages),
    )

    messages = [
        {"role": "system", "content": system_task},
        {"role": "user", "content": user_task},
    ]
    return messages
