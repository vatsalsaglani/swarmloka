import json
from ._types import *
from ..llm import BaseLLM
from copy import deepcopy
from .prompts import FINAL_ANSWER_PROMPT
import asyncio


class Loka:

    def __init__(self,
                 orchestrator_agent: Agent,
                 swarm: List[Agent],
                 llm: BaseLLM,
                 messages: List[Dict],
                 max_iterations: int = 10):
        self.orchestrator_agent = orchestrator_agent
        self.llm = llm
        self.swarm = swarm.copy(
        )  # agents are copied to avoid mutating the original list
        self.orchestrator_agent.functions = [
            AgentFunction(name=a.name,
                          description=a.instruction,
                          parameters={
                              "properties": {
                                  "agent_name": {
                                      "type":
                                      "string",
                                      "description":
                                      "The name of the agent to call"
                                  }
                              }
                          },
                          _callable=a.name) for a in swarm
        ]
        self.orchestrator_agent.functions.append(
            AgentFunction(name="end",
                          description="Use this when reached the end",
                          parameters=End.model_json_schema(),
                          _callable="end"))
        self.loka_map = {f.name: f for f in self.orchestrator_agent.functions}
        self.swarm_map = {a.name: a for a in self.swarm}
        self.messages = messages
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.selected_agents = []

    async def _iterate(self, model_name: str, llm_args: Optional[Dict] = {}):
        curr_messages = deepcopy(self.messages)
        functions = list(
            map(
                lambda x: {
                    "name": x.name,
                    "description": x.description,
                    "parameters": x.parameters
                }, self.orchestrator_agent.functions))
        response_functions = await self.llm.function_call(
            messages=curr_messages,
            model_name=model_name,
            functions=functions,
            validate_params=False,
            **llm_args)
        return response_functions

    async def _function_args(self,
                             model_name: str,
                             agent_name: str,
                             no_func_call: bool = False,
                             max_rec_depth: int = 3,
                             curr_depth: int = 0,
                             llm_args: Optional[Dict] = {}):
        function = self.swarm_map[agent_name]
        function_list = list(
            map(
                lambda x: {
                    "name": x.get("name"),
                    "description": x.get("description"),
                    "parameters": x.get("parameters")
                }, function.functions))
        curr_messages = deepcopy(self.messages)
        if no_func_call:
            curr_messages[-1][
                "content"] += f"\n\nDidn't receive a function. Please return only one appropriate function."
        response = await self.llm.function_call(curr_messages,
                                                model_name,
                                                function_list,
                                                validate_params=True,
                                                **llm_args)
        if len(response) == 0:
            curr_depth += 1
            if curr_depth > max_rec_depth:
                raise Exception(
                    f"Maximum recursion depth reached while waiting for function call from Swarm: {agent_name}"
                )
            return await self._function_args(model_name, agent_name, True,
                                             max_rec_depth, curr_depth,
                                             llm_args)
        return response[-1]

    async def _return_final_answer(self, model_name: str, end_output: Dict):
        self.messages[-1]["content"] += f'\n\n{end_output.get("content")}'
        final_messages = [{
            "role": "system",
            "content": FINAL_ANSWER_PROMPT.PROMPT
        }] + [{
            "role":
            "user",
            "content":
            f'The following is the list of agents and functions used to solve the problem:\n{self.selected_agents}'
        }]
        # print("=====[ANSWER]=====\n")
        async for chunk in self.llm.stream_complete(final_messages,
                                                    model_name):
            yield chunk

    async def swarmloka(self, model_name: str, llm_args: Optional[Dict] = {}):
        done = False
        while self.current_iteration < self.max_iterations + 1 and not done:
            desired_swarm = await self._iterate(model_name, llm_args)
            yield {"desired_swarm": desired_swarm}
            yield "\n\n"
            self.selected_agents.append(
                {"agent": desired_swarm[0].get("name")})
            func_op = {"role": "assistant", "content": ""}
            for swarm in desired_swarm:
                if swarm.get("name") == "end":
                    done = swarm.get("parameters")
                    break
                if not swarm.get("name") in self.swarm_map:
                    raise Exception(
                        f"Swarm: {swarm.get('name')} is not part of the Loka")
                else:
                    func_op[
                        "content"] += f"Selected Agent: {swarm.get('name')}\n"
                    function_args = await self._function_args(
                        model_name,
                        swarm.get("name"),
                        curr_depth=0,
                        llm_args=llm_args)
                    yield {"function_args": function_args}
                    yield "\n\n"
                    if not "functions" in self.selected_agents[-1]:
                        self.selected_agents[-1]["functions"] = []
                    self.selected_agents[-1]["functions"].append({
                        "name":
                        function_args.get("name"),
                        "parameters":
                        function_args.get("parameters")
                    })
                    func_op[
                        "content"] += f"Function Name: {function_args.get('name')}\n"
                    func_op[
                        "content"] += f"Function Args: {function_args.get('parameters')}\n"
                    agent_functions = self.swarm_map[swarm.get(
                        "name")].functions
                    agent_function = next(
                        (f for f in agent_functions
                         if f.get("name") == function_args.get("name")), None)
                    if agent_function is None:
                        raise Exception(
                            f"Function: {function_args.get('name')} is not part of the Swarm: {swarm.get('name')}"
                        )
                    if "_callable" in agent_function and callable(
                            agent_function.get("_callable")):
                        func = agent_function.get("_callable")
                        params = function_args.get("parameters", {})
                        if asyncio.iscoroutinefunction(func):
                            results = await func(**params)
                            if hasattr(results, '__aiter__'):
                                collected_results = []
                                async for chunk in results:
                                    collected_results.append(chunk)
                                    print(chunk, end="", flush=True)
                                print("\n\n")
                                results = collected_results
                        else:
                            results = func(**params)
                            if hasattr(results, '__iter__') and not isinstance(
                                    results, (str, bytes, dict)):
                                collected_results = []
                                for chunk in results:
                                    collected_results.append(chunk)
                                    print(chunk, end="", flush=True)
                                print("\n\n")
                                results = collected_results
                    else:
                        results = function_args.get("parameters")
                    self.selected_agents[-1]["functions"][-1][
                        "result"] = results
                    yield {"function_result": results}
                    yield "\n\n"
                    func_op["content"] += f"Function Result: {results}\n"
                    self.messages[-1]["content"] += f"\n\n{func_op['content']}"
            self.current_iteration += 1
        if done:
            yield {"final_answer": done}
            yield "\n\n"
            yield "\n\n"
            async for chunk in self._return_final_answer(model_name, done):
                yield chunk
