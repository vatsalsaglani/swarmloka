import json
from ._types import *
from ..llm import BaseLLM
from copy import deepcopy
from .prompts import FINAL_ANSWER_PROMPT
import asyncio
from rich.console import Console
from rich.status import Status


class Loka:

    def __init__(self,
                 orchestrator_agent: Agent,
                 swarm: List[Agent],
                 llm: BaseLLM,
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
        # self.messages = messages
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.selected_agents = []
        self.working_output_memory = {}

    # TODO: Optimize this to be more efficient
    def _update_working_output_memory(self):
        # print(json.dumps(self.selected_agents, indent=2))
        for ix, swarm in enumerate(self.selected_agents):
            prefix = f"swarm_{ix}"
            for ifx, function in enumerate(swarm.get('functions', [])):
                self.working_output_memory[
                    f"{prefix}_function_{function['name']}_{ifx}"] = function[
                        'result']

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

    def _parse_function_parameters(self, parameters: Dict):

        def parse_value(value):
            if isinstance(value, str) and value.startswith("swarm_"):
                if value in self.working_output_memory:
                    return self.working_output_memory[value]
                raise KeyError(
                    f"Key {value} is not in the working output memory.")
            elif isinstance(value, (list, tuple)):
                return type(value)([parse_value(v) for v in value])
            elif isinstance(value, dict):
                return {k: parse_value(v) for k, v in value.items()}
            return value

        return {key: parse_value(value) for key, value in parameters.items()}

    # def _format_working_output_memory(self):

    async def _verify_input_available_in_working_memory(
            self,
            model_name: str,
            agent_name: str,
            llm_args: Optional[Dict] = {}):

        class RequiredKeyValue(BaseModel):
            key: str
            value: str

        class VerifyAnyRequiredKeyAvailableInWorkingMemory(BaseModel):
            required_keys: Union[List[RequiredKeyValue], None]

        required_key_function = [{
            "name":
            "verify_any_required_key_available_in_working_memory",
            "description":
            "Verify if any of the required keys are available in the working memory",
            "parameters":
            VerifyAnyRequiredKeyAvailableInWorkingMemory.model_json_schema()
        }]
        function = self.swarm_map[agent_name]
        function_list = list(
            map(
                lambda x: {
                    "name": x.get("name"),
                    "description": x.get("description"),
                    "parameters": x.get("parameters")
                }, function.functions))
        system_message = f"""
        You are an expert at verifying if any of the required keys are available in the working memory. Your are provided with the working memory and a set of agents that are already called with their results.
        You are also provided with the current agent that needs to be called with required parameters. You need to check if any of the required keys for the current agent are available in the working memory and list them out.
        """
        user_message = f"""
        Working Memory: {self.working_output_memory}
        Called Agents: {self.selected_agents}
        Current Agent: {function_list}
        """
        messages = [{
            "role": "system",
            "content": system_message
        }, {
            "role": "user",
            "content": user_message
        }]
        response = await self.llm.function_call(messages, model_name,
                                                required_key_function,
                                                **llm_args)
        return response[-1].get("parameters")

    async def _function_args(self,
                             model_name: str,
                             agent_name: str,
                             no_func_call: bool = False,
                             max_rec_depth: int = 3,
                             curr_depth: int = 0,
                             llm_args: Optional[Dict] = {}):
        required_keys_available_in_working_memory = await self._verify_input_available_in_working_memory(
            model_name, agent_name, llm_args)
        print(
            f"REQUIRED KEYS AVAILABLE IN WORKING MEMORY: {required_keys_available_in_working_memory}"
        )
        function = self.swarm_map[agent_name]
        function_list = list(
            map(
                lambda x: {
                    "name": x.get("name"),
                    "description": x.get("description"),
                    "parameters": x.get("parameters")
                }, function.functions))
        curr_messages = deepcopy(self.messages)
        curr_messages[-1][
            "content"] += f"""\n\nBelow is the context of the working output memory in double backticks:
            ``{self.working_output_memory}``

            When calling your next function, reference any previous output by simply providing the dictionary key in the parameters, e.g.

            <functioncall>{{
                "name": "someFunction", 
                "parameters": {{
                    "previousOutput": "swarm_1_function_create_sql_query_0"
                    }}
                }}
            </functioncall>

            Make sure to only reference keys that exist in the working output memory.
            """
        if required_keys_available_in_working_memory:
            curr_messages[-1][
                "content"] += f"\n\nThe following keys are already available in the working memory: {required_keys_available_in_working_memory}. Please use these keys in the parameters."
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
        try:
            response = response[-1]
            print(f"RESPONSE PRE PARSE: {json.dumps(response, indent=2)}")
            response["parameters"] = self._parse_function_parameters(
                response["parameters"])
            print(f"RESPONSE POST PARSE: {json.dumps(response, indent=2)}")
        except KeyError as error:
            curr_depth += 1
            if curr_depth > max_rec_depth:
                raise Exception(
                    f"Maximum recursion depth reached while waiting for function call from Swarm: {agent_name}"
                )
            curr_messages[-1]["content"] += f"\n\n{error}"
            return await self._function_args(model_name, agent_name, True,
                                             max_rec_depth, curr_depth,
                                             llm_args)
        return response

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

    async def swarmloka(self,
                        model_name: str,
                        messages: List[Dict],
                        write_end_result: bool = True,
                        llm_args: Optional[Dict] = {}):
        done = False
        console = Console()
        self.messages = messages
        self.current_iteration = 0
        while self.current_iteration < self.max_iterations + 1 and not done:
            with Status("[bold blue]🤖 Orchestrator thinking...",
                        console=console) as status:
                desired_swarm = await self._iterate(model_name, llm_args)
            yield {"desired_swarm": desired_swarm}
            yield "\n\n"
            status.update(
                f"[bold green]🤖 Orchestrator selected swarm: {desired_swarm[0].get('name')}"
            )
            self.selected_agents.append(
                {"agent": desired_swarm[0].get("name")})
            func_op = {"role": "assistant", "content": ""}

            for swarm in desired_swarm:
                if swarm.get("name") == "end":
                    done = swarm.get("parameters")
                    # print(done)
                    reason = done.get("why")
                    status.update(
                        f"[bold green]🤖 Orchestrator reached end: {reason}")
                    break

                if not swarm.get("name") in self.swarm_map:
                    raise Exception(
                        f"Swarm: {swarm.get('name')} is not part of the Loka")
                else:
                    swarm_name = swarm.get('name')
                    status_msg = f"[bold blue]🔄 Swarm '{swarm_name}' processing..."

                    with Status(status_msg, console=console) as status:
                        func_op["content"] += f"Selected Agent: {swarm_name}\n"
                        function_args = await self._function_args(
                            model_name,
                            swarm_name,
                            curr_depth=0,
                            llm_args=llm_args)

                        status.update(
                            f"[bold green]🤖 Executing {swarm_name} → {function_args.get('name')}..."
                        )

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

                        agent_functions = self.swarm_map[swarm_name].functions
                        agent_function = next(
                            (f for f in agent_functions
                             if f.get("name") == function_args.get("name")),
                            None)

                        if agent_function is None:
                            raise Exception(
                                f"Function: {function_args.get('name')} is not part of the Swarm: {swarm_name}"
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
                                    results = collected_results
                            else:
                                results = func(**params)
                                if hasattr(results,
                                           '__iter__') and not isinstance(
                                               results, (str, bytes, dict)):
                                    collected_results = []
                                    for chunk in results:
                                        collected_results.append(chunk)
                                    results = collected_results
                        else:
                            results = function_args.get("parameters")

                        self.selected_agents[-1]["functions"][-1][
                            "result"] = results
                        yield {"function_result": results}
                        yield "\n\n"

                        func_op["content"] += f"Function Result: {results}\n"
                        self.messages[-1][
                            "content"] += f"\n\n{func_op['content']}"

                        self._update_working_output_memory()

            self.current_iteration += 1

        if done:
            if write_end_result:
                console.print(
                    f"[bold green]✨ Generating final answer...[/bold green]")
                yield {"final_answer": done}
                yield "\n\n"
                async for chunk in self._return_final_answer(model_name, done):
                    if isinstance(chunk, dict):
                        console.print(f"[cyan]{chunk}[/cyan]")
                    elif isinstance(chunk, str):
                        console.print(f"[cyan]{chunk}[/cyan]", end="")
                    else:
                        console.print(f"[yellow]{chunk}[/yellow]")
                    yield chunk
