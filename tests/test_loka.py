import asyncio
from pydantic import BaseModel, Field
from typing import List
from swarmloka import Loka, LocalLLM
from swarmloka import Agent, AgentFunction


class Multiply(BaseModel):
    numbers: List[int] = Field(description="List of numbers to multiply")


def multiplication_agent(numbers: List[int]):
    result = 1
    for num in numbers:
        result *= num
    return result


class Add(BaseModel):
    numbers: List[int] = Field(description="List of numbers to add")


def addition_agent(numbers: List[int]):
    return sum(numbers)


local_multiply_agent = Agent(name="multiply",
                             instruction="Multiply the given numbers",
                             functions=[
                                 dict(name="multiply",
                                      description="Multiply the given numbers",
                                      parameters=Multiply.model_json_schema(),
                                      _callable=multiplication_agent)
                             ])

local_add_agent = Agent(name="add",
                        instruction="Add the given numbers",
                        functions=[
                            dict(name="add",
                                 description="Add the given numbers",
                                 parameters=Add.model_json_schema(),
                                 _callable=addition_agent)
                        ])

# llm = LocalLLM("http://localhost:1234/v1", "api-key")
model_name = "gpt-4o"
llm = LocalLLM(
    "https://api.openai.com/v1",
    "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"
)

local_swarm = Loka(orchestrator_agent=Agent(
    name="orchestrator",
    instruction="Orchestrate the given numbers",
    functions=[]),
                   swarm=[local_multiply_agent, local_add_agent],
                   llm=llm,
                   max_iterations=10)


async def main():
    collected_chunks = []
    async for chunk in local_swarm.swarmloka(
            model_name=model_name,
            messages=[{
                "role":
                "user",
                "content":
                "Multiply 2 and 3 and then add 4 to the result"
            }],
            write_end_result=True,
            llm_args={"temperature": 0.2}):
        collected_chunks.append(chunk)
        # print(chunk, end="", flush=True)


asyncio.run(main())
