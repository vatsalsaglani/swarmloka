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
    exit_here: bool = False


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


class OrchestratorAgentThinking(BaseModel):
    observe: Union[List[str], None] = Field(
        ...,
        description=
        "List of Observations of the environment and what's completed. Write your obeservation from all user and assistant interactions."
    )
    think: Union[List[str], None] = Field(
        ...,
        description=
        "List of Thoughts on what to do next. Write your thoughts based on the observations from all user and assistant interactions."
    )
    action: Union[str, None] = Field(
        ...,
        description=
        "Action to take. Write your action based on the thoughts from all user and assistant interactions. Action should be a valid agent name."
    )
    agent_name: str = Field(...,
                            description="Valid agent name based on the action")
