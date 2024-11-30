import json
from ._types import *
from ..llm import BaseLLM
from copy import deepcopy


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
        final_answer_system_prompt = """You are a helpful AI assistant that explains the problem-solving process to users. You will receive:
            1. The original user request
            2. A list of steps taken by different agents
            3. The final output

            Your task is to create a clear, conversational explanation that helps users understand:
            - What their request was
            - Which agents were involved
            - What functions were used and why
            - The intermediate results
            - The final answer

            Format your response as follows:
            1. Brief restatement of what the user asked for
            2. Step-by-step breakdown of what happened
            3. Final result with explanation

            Example:

            Input:
            User Request: "Check temperature in London and notify if above 20C"
            Steps: [
                {"agent": "weather_agent", "function": "getWeather", "params": {"city": "London"}, "output": "22C"},
                {"agent": "notification_agent", "function": "sendNotification", "params": {"message": "Temperature alert"}, "output": "Notification sent"}
            ]
            Final Output: {"why": "Temperature check and notification complete", "final_answer": "Temperature was 22C, notification sent"}

            Response:
            I helped you check London's temperature and send a notification since it was above 20C. Here's what happened:

            1. Temperature Check:
            • Weather_agent checked London's current temperature
            • The temperature was 22°C

            2. Notification:
            • Since 22°C was above your threshold of 20°C
            • Notification_agent sent out an alert

            The process is complete - London is at 22°C and you've been notified.

            Remember to:
            - Use clear, conversational language
            - Break down complex steps into digestible pieces
            - Show the logical flow from one step to the next
            - Highlight key numbers and results
            - Explain why each step was necessary
            - Make technical processes understandable to non-technical users"""
        self.messages[-1]["content"] += f'\n\n{end_output.get("content")}'
        final_messages = [{
            "role": "system",
            "content": final_answer_system_prompt
        }] + [{
            "role":
            "user",
            "content":
            f'The following is the list of agents and functions used to solve the problem:\n{self.selected_agents}'
        }]
        print("=====[ANSWER]=====\n")
        async for chunk in self.llm.stream_complete(final_messages,
                                                    model_name):
            yield chunk

    async def swarmloka(self, model_name: str, llm_args: Optional[Dict] = {}):
        done = False
        while self.current_iteration < self.max_iterations + 1 and not done:
            desired_swarm = await self._iterate(model_name, llm_args)
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
                        results = agent_function.get("_callable")(
                            **function_args.get("parameters", {}))
                    else:
                        results = function_args.get("parameters")
                    self.selected_agents[-1]["functions"][-1][
                        "result"] = results
                    func_op["content"] += f"Function Result: {results}\n"
                    self.messages[-1]["content"] += f"\n\n{func_op['content']}"
            self.current_iteration += 1
        if done:
            async for chunk in self._return_final_answer(model_name, done):
                print(chunk, end="", flush=True)
        return self.messages, self.selected_agents
