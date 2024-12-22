from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Tuple, Callable, Union, Any


class AgentFunction(BaseModel):
    name: str
    description: str
    parameters: Dict
    _callable: Any

    class Config:
        arbitrary_types_allowed = True


class Agent(BaseModel):
    name: str
    instruction: str
    functions: List[Dict[str, Any]]


class End(BaseModel):
    end: bool = Field(default=True,
                      description="Use this when reached the end")
    why: str = Field(
        ..., description="Write the reason for ending the conversation")
    final_answer: str = Field(
        ..., description="Write the final answer and the steps taken")


class ContextVariable(BaseModel):
    ctx_key: str = Field(..., description="Context key from working memory")
    description: str = Field(...,
                             description="Description of the context variable")
    content_type: str = Field(
        ...,
        description=
        "Type of the content (e.g., 'list', 'dict', 'string', 'List[Dict]')")
