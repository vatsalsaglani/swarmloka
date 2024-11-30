# Swarmloka

Swarmloka is a Python framework for orchestrating local agent swarms with LLM-based reasoning. It enables multiple agents to collaborate and solve complex tasks using a powerful, local-first approach with models like Qwen 2.5.

## Features
- Local-first agent collaboration.
- Support for orchestrating complex workflows.
- Integration with LLM servers like Qwen 2.5, llama.cpp, and LM Studio.
- Extensible and customizable agent functions.

---

## Getting Started

### Prerequisites
- Python 3.12+
- Poetry for dependency management
- One of the following LLM servers running locally:
  - [LM Studio Server](https://lmstudio.ai/) (recommended)
  - llama.cpp server
  - Ollama server

---

### Installation
Clone the repository and install dependencies:

```
pip install git+https://github.com/vatsalsaglani/swarmloka.git
```

---

### Setup Your LLM Server
1. Download the **LM Studio Server** from [here](https://lmstudio.ai/) and follow the setup instructions.
2. Search for the **Qwen-2.5-Coder** model in the model list.
3. Load the model and start the server.

---

### Usage Example
Here's how to use `Swarmloka` to orchestrate a simple swarm:

```python
import asyncio
from pydantic import BaseModel, Field
from typing import List
from swarmloka import Loka, LocalLLM
from swarmloka import Agent, AgentFunction

# Define agent tasks
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

# Define agents
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

# Initialize LLM and Swarm
llm = LocalLLM("http://localhost:1234/v1", "api-key")

local_swarm = Loka(orchestrator_agent=Agent(
    name="orchestrator",
    instruction="Orchestrate the given numbers",
    functions=[]),
                   swarm=[local_multiply_agent, local_add_agent],
                   llm=llm,
                   messages=[{
                       "role": "user",
                       "content": "Multiply 2 and 3 and then add 4 to the result"
                   }],
                   max_iterations=10)

# Run the swarm
print(
    asyncio.run(
        local_swarm.swarmloka("qwen2.5-coder-3b-instruct-q4_k_m",
                              {"temperature": 0.2})))
```

---


### Supported Models
The system is designed to work with models like:
- [Qwen-2.5-Coder-3B-Instruct](https://huggingface.co/lmstudio-community/Qwen2.5-Coder-3B-Instruct-GGUF)

---

### Future Work
- Adding more agents and workflows.
- Support for additional LLM servers and models.
- Optimization for large-scale tasks.

---

### License
This project is licensed under the MIT License. See `LICENSE` for details.

---

### Contributions
Contributions are welcome!
---

### Acknowledgments
- Powered by Qwen 2.5 and local-first AI technology.
- Inspired by OpenAI's experimental Swarm framework.

---

Enjoy orchestrating with Swarmloka! 🚀