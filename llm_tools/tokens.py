from dataclasses import dataclass
from typing import List, Dict
import logging
import tiktoken


from llm_tools.chat_message import (
    OpenAIChatMessage,
    prepare_messages,
    convert_message_to_dict,
)

logger = logging.getLogger(__name__)


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

    def price_per_1e6_input_tokens(self) -> int:
        return {
            "gpt-3.5-turbo": 2,
            "gpt-4": 30,
            "gpt-4-1106-preview": 10,
        }[self.model_name]

    def price_per_1e6_output_tokens(self) -> int:
        return {
            "gpt-3.5-turbo": 2,
            "gpt-4": 60,
            "gpt-4-1106-preview": 30,
        }[self.model_name]

    def get_price_multiplied_by_1e6(self) -> int:
        return (
            self.price_per_1e6_input_tokens() * self.n_input_tokens
            + self.price_per_1e6_output_tokens() * self.n_output_tokens
        )

    def get_price(self) -> float:
        return self.get_price_multiplied_by_1e6() / 1e6


@dataclass
class TokenExpenses:
    expenses: Dict[str, TokenExpense]

    def __init__(self):
        self.expenses = {}

    def add_expense(self, expense: TokenExpense):
        if expense.model_name in self.expenses:
            self.expenses[expense.model_name] += expense
        else:
            self.expenses[expense.model_name] = expense

    def __add__(self, other: "TokenExpenses") -> "TokenExpenses":
        result = TokenExpenses()
        for expense in self.expenses.values():
            result.add_expense(expense)
        for expense in other.expenses.values():
            result.add_expense(expense)
        return result

    def get_price_multiplied_by_1e6(self) -> int:
        return sum(expense.get_price_multiplied_by_1e6() for expense in self.expenses.values())

    def get_price(self) -> float:
        return self.get_price_multiplied_by_1e6() / 1e6


def count_tokens_from_input_messages(
    messages: List[OpenAIChatMessage],
    model_name: str,
) -> int:
    if not messages:
        return 0
    messages_typed = prepare_messages(messages)
    message_dicts = [convert_message_to_dict(x) for x in messages_typed]
    return num_tokens_from_messages(
        messages=message_dicts,
        model=model_name,
    )


def count_tokens_from_output_text(
    text: str,
    model_name: str,
) -> int:
    if not text:
        return 0
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.debug("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
