from typing import (
    Dict,
    Union,
    List,
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatMessage
from funcy import omit


OpenAIChatMessage = Union[BaseMessage, Dict[str, str]]



def convert_dict_to_message(_dict: Dict[str, str]) -> BaseMessage:
    role = _dict["role"]
    additional_kwargs = dict(omit(_dict, ["role", "content"]))
    if role == "user":
        return HumanMessage(content=_dict["content"], additional_kwargs=additional_kwargs)
    elif role == "assistant":
        return AIMessage(content=_dict["content"], additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"], additional_kwargs=additional_kwargs)
    else:
        return ChatMessage(content=_dict["content"], role=role, additional_kwargs=additional_kwargs)
    

def prepare_message(message: OpenAIChatMessage) -> BaseMessage:
    if isinstance(message, dict):
        return convert_dict_to_message(message)
    elif isinstance(message, BaseMessage):
        return message
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def prepare_messages(messages: List[OpenAIChatMessage]) -> List[BaseMessage]:
    return [prepare_message(message) for message in messages]
