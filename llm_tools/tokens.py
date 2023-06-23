from dataclasses import dataclass
from typing import List, Dict

from tiktoken import encoding_for_model
from llm_tools.chat_message import (
    OpenAIChatMessage,
    prepare_messages,
    convert_message_to_dict,
)


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
            "gpt-3.5-turbo-16k": 3,
            "gpt-4": 30,
        }[self.model_name]
    
    def price_per_1e6_output_tokens(self) -> int:
        return {
            "gpt-3.5-turbo": 2,
            "gpt-3.5-turbo-16k": 4,
            "gpt-4": 60,
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
    encoding = encoding_for_model(model_name)
    messages_typed = prepare_messages(messages)
    message_dicts = [convert_message_to_dict(x) for x in messages_typed]

    if model_name == "gpt-3.5-turbo":
        tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model_name == "gpt-3.5-turbo-16k":
        tokens_per_message = 4 
        tokens_per_name = -1
    elif model_name == "gpt-4":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_input_tokens = 0
    for message_dict in message_dicts:
        n_input_tokens += tokens_per_message
        for key, value in message_dict.items():
            n_input_tokens += len(encoding.encode(value))
            if key == "name":
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
