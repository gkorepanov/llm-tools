from dataclasses import dataclass
from typing import List

from tiktoken import encoding_for_model
from llm_tools.chat_message import OpenAIChatMessage, prepare_messages


@dataclass
class TokenExpense:
    model_name: str
    n_input_tokens: int = 0
    n_output_tokens: int = 0

    @property
    def n_total_tokens(self) -> int:
        return self.n_input_tokens + self.n_output_tokens
    
    def __add__(self, other: "TokenExpense") -> "TokenExpense":
        if other.model_name != self.model_name:
            raise ValueError("Cannot add TokenExpense objects with different model names")
        return TokenExpense(
            model_name=self.model_name,
            n_input_tokens=self.n_input_tokens + other.n_input_tokens,
            n_output_tokens=self.n_output_tokens + other.n_output_tokens,
        )


def count_tokens_from_input_messages(
    messages: List[OpenAIChatMessage],
    model_name: str,
) -> int:
    if not messages:
        return 0
    encoding = encoding_for_model(model_name)
    messages_typed = prepare_messages(messages)

    if model_name == "gpt-3.5-turbo":
        tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model_name == "gpt-4":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_input_tokens = 0
    for message in messages_typed:
        n_input_tokens += tokens_per_message
        for message in messages_typed:
            n_input_tokens += len(encoding.encode(message.content))
            if "name" in message.additional_kwargs:
                n_input_tokens += tokens_per_name

    n_input_tokens += 3
    return n_input_tokens


def count_tokens_from_output_text(
    text: str,
    model_name: str,
) -> int:
    if not text:
        return 0
    encoding = encoding_for_model(model_name)
    return len(encoding.encode(text))
