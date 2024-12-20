import asyncio
from pydantic import BaseModel, Field
from typing import List
from swarmloka import Loka, GeminiLLM, LocalLLM
from swarmloka import Agent


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


gemini_multiply_agent = Agent(name="multiply",
                              instruction="Multiply the given numbers",
                              functions=[
                                  dict(
                                      name="multiply",
                                      description="Multiply the given numbers",
                                      parameters=Multiply.model_json_schema(),
                                      _callable=multiplication_agent)
                              ])

gemini_add_agent = Agent(name="add",
                         instruction="Add the given numbers",
                         functions=[
                             dict(name="add",
                                  description="Add the given numbers",
                                  parameters=Add.model_json_schema(),
                                  _callable=addition_agent)
                         ])
llm = LocalLLM("https://generativelanguage.googleapis.com/v1beta/openai",
               "YOUR_API_KEY")

swarm = Loka(orchestrator_agent=Agent(
    name="orchestrator",
    instruction="Orchestrate the given numbers",
    functions=[]),
             swarm=[gemini_multiply_agent, gemini_add_agent],
             llm=llm,
             messages=[{
                 "role":
                 "user",
                 "content":
                 "Multiply 2 and 3 and then add 4 to the result"
             }],
             max_iterations=10)


async def main():
    async for chunk in swarm.swarmloka("gemini-2.0-flash-exp",
                                       {"temperature": 0.2}):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
