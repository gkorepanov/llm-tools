from dataclasses import dataclass
from typing import Dict
import logging

from litellm import get_model_info


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
        return int(get_model_info(self.model_name)['input_cost_per_token'] * 1e6)

    def price_per_1e6_output_tokens(self) -> int:
        return int(get_model_info(self.model_name)['output_cost_per_token'] * 1e6)

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
