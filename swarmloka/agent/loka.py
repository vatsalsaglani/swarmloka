import traceback
from collections.abc import AsyncGenerator
import json
from ._types import *
from ..llm import BaseLLM
from copy import deepcopy
from .prompts import FINAL_ANSWER_PROMPT
import asyncio
from rich.console import Console
from rich.status import Status


class Loka:
    """A class to orchestrate swarm-based agent interactions and function execution.
    The Loka class manages a swarm of agents, coordinating their interactions through
    an orchestrator agent. It handles function execution, memory management, and 
    result generation in an asynchronous environment.
    Attributes:
        orchestrator_agent (Agent): The main agent that coordinates the swarm
        llm (BaseLLM): Language model interface for agent communication
        swarm (List[Agent]): List of available agents in the swarm
        max_iterations (int): Maximum number of interaction iterations
        current_iteration (int): Current iteration count
        selected_agents (List): History of selected agents and their actions
        working_output_memory (Dict): Memory store for function outputs
    Example:
        >>> orchestrator = Agent(name="orchestrator", instruction="Coordinate agents")
        >>> swarm = [Agent(name="agent1"), Agent(name="agent2")]
        >>> loka = Loka(orchestrator, swarm, llm_instance)
        >>> async for result in loka.swarmloka("gpt-4", messages):
        ...     print(result)
    """

    def __init__(self,
                 orchestrator_agent: Agent,
                 swarm: List[Agent],
                 llm: BaseLLM,
                 max_iterations: int = 10,
                 verbose: bool = False):
        """Initialize the Loka instance with required components.

        Args:
            orchestrator_agent (Agent): The main coordinating agent
            swarm (List[Agent]): List of available agents
            llm (BaseLLM): Language model interface
            max_iterations (int, optional): Maximum iterations. Defaults to 10.

        Note:
            The initialization process:
            1. Copies the swarm to avoid mutating the original list
            2. Sets up orchestrator agent functions based on swarm agents
            3. Initializes working memory and tracking variables
        """
        self.orchestrator_agent = orchestrator_agent
        self.llm = llm
        self.swarm = swarm.copy()
        self.console = Console()  # Initialize console here

        self.orchestrator_agent.functions = [
            AgentFunction(
                name=a.name,
                description=a.instruction,
                parameters=OrchestratorAgentThinking.model_json_schema(),
                #   parameters={
                #       "properties": {
                #           "agent_name": {
                #               "type":
                #               "string",
                #               "description":
                #               "The name of the agent to call"
                #           }
                #       }
                #   },
                _callable=a.name) for a in swarm
        ]
        self.orchestrator_agent.functions.append(
            AgentFunction(name="end",
                          description="Use this when reached the end",
                          parameters=End.model_json_schema(),
                          _callable="end"))
        self.loka_map = {f.name: f for f in self.orchestrator_agent.functions}
        self.swarm_map = {a.name: a for a in self.swarm}
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.selected_agents = []
        self.working_output_memory = {}
        self.verbose = verbose

    def _update_working_output_memory(self):
        """Update the working memory with results from selected agents' functions.

        This method processes the selected_agents list and updates working_output_memory
        with the results of each function execution. The memory keys are formatted as:
        'swarm_{agent_index}_function_{function_name}_{function_index}'

        Example:
            If agent 0 executed a function named 'analyze' as its first function:
            >>> self._update_working_output_memory()
            # Updates working_output_memory with:
            # {'swarm_0_function_analyze_0': <function_result>}
        """
        for ix, swarm in enumerate(self.selected_agents):
            prefix = f"swarm_{ix}"
            for ifx, function in enumerate(swarm.get('functions', [])):
                self.working_output_memory[
                    f"{prefix}_function_{function['name']}_{ifx}"] = function[
                        'result']

    async def _iterate(self,
                       model_name: str,
                       llm_args: Optional[Dict] = {},
                       retry_count: int = 0):
        """Execute one iteration of the orchestrator's decision-making process.

        This method sends the current context to the LLM and gets the next action
        to be taken by the swarm. Includes retry logic for empty responses.

        Args:
            model_name (str): Name of the LLM model to use
            llm_args (Optional[Dict], optional): Additional LLM parameters. Defaults to {}.
            retry_count (int, optional): Current retry attempt. Defaults to 0.

        Returns:
            Dict: The LLM's function call response indicating the next action

        Raises:
            Exception: If max retries reached without valid response

        Example:
            >>> response = await self._iterate("gpt-4", {"temperature": 0.7})
            >>> print(response)
            {'name': 'agent1', 'parameters': {...}}
        """
        MAX_RETRIES = 3
        curr_messages = deepcopy(self.messages)

        curr_messages = [{
            "role": "system",
            "content": self.orchestrator_agent.instruction
        }] + curr_messages

        if retry_count > 0:
            curr_messages.append({
                "role":
                "user",
                "content":
                f"""Previous attempt didn't select any agent. You MUST select one of the available agents to proceed.
                
Your role: {self.orchestrator_agent.instruction}

Remember:
1. You must choose an agent
2. Select the most appropriate agent for the current task
3. Or use 'end' if the task is complete"""
            })

        if self.verbose:
            self.console.print(
                f'Messages Iterate: {json.dumps(curr_messages, indent=2)}')
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
            validate_params=True,
            **llm_args)

        if self.verbose:
            self.console.print(
                f"Iterate Response Functions: {json.dumps(response_functions, indent=2)}"
            )

        if not response_functions:
            if retry_count >= MAX_RETRIES:
                raise Exception(
                    f"Failed to get valid agent selection after {MAX_RETRIES} attempts"
                )

            self.console.print(
                f"Empty response received. Attempt {retry_count + 1}/{MAX_RETRIES}"
            )
            return await self._iterate(model_name=model_name,
                                       llm_args=llm_args,
                                       retry_count=retry_count + 1)

        return response_functions

    def _parse_function_parameters(self, parameters: Dict):
        """Parse and resolve function parameters with memory references.

        This method processes function parameters and resolves any references to
        working memory, allowing functions to use results from previous operations.

        Args:
            parameters (Dict): Raw function parameters that may contain memory references

        Returns:
            Dict: Processed parameters with resolved memory references

        Example:
            >>> params = {"input": "swarm_0_function_result_0"}
            >>> resolved = self._parse_function_parameters(params)
            # If the memory reference exists, it's replaced with the actual value
        """

        def parse_value(value):
            """Recursively parse and resolve memory references in parameter values.
            
            Args:
                value: The value to parse, can be str, dict, list, or other types
            
            Returns:
                The resolved value from memory if it's a reference, otherwise the original value
            
            Example:
                >>> parse_value("swarm_0_function_result_0")
                # Returns the actual value from working_output_memory if it exists
            """
            if isinstance(value, str) and value.startswith("swarm_"):
                if value in self.working_output_memory:
                    return self.working_output_memory[value]
                raise KeyError(
                    f"Key {value} is not in the working output memory.")

            if isinstance(value, dict) and "ctx_key" in value:
                ctx_key = value["ctx_key"]
                if ctx_key in self.working_output_memory:
                    return self.working_output_memory[ctx_key]
                raise KeyError(
                    f"Context key {ctx_key} is not in the working output memory."
                )

            elif isinstance(value, (list, tuple)):
                return type(value)([parse_value(v) for v in value])
            elif isinstance(value, dict):
                return {k: parse_value(v) for k, v in value.items()}
            return value

        return {key: parse_value(value) for key, value in parameters.items()}

    async def _verify_input_available_in_working_memory(
            self,
            model_name: str,
            agent_name: str,
            llm_args: Optional[Dict] = {}):
        """Verify if required function inputs are available in working memory.

        This method analyzes the working memory and the required parameters for
        the agent's functions to identify available and suitable memory entries.

        Args:
            model_name (str): Name of the LLM model to use
            agent_name (str): Name of the agent whose functions need verification
            llm_args (Optional[Dict], optional): Additional LLM parameters. Defaults to {}.

        Returns:
            Dict: Analysis results including:
                - analysis: Chain-of-thought reasoning about memory availability
                - required_keys: List of memory keys that can be used for parameters

        Example:
            >>> result = await self._verify_input_available_in_working_memory(
            ...     "gpt-4", "agent1", {})
            >>> print(result["analysis"])
            "Found matching memory entries for required parameters..."
        """

        class RequiredKeyValue(BaseModel):
            parameter_key: str = Field(
                description=
                "The parameter name required for the agent functions")
            value_key: str = Field(
                description="The key in working memory that can be used")
            reasoning: str = Field(
                description=
                "Detailed explanation of why this memory key is suitable")
            content_type: str = Field(
                description="Type of content (e.g., list, dict, string, number)"
            )
            similarity_score: float = Field(
                description="How relevant/similar the content is (0-1)",
                ge=0,
                le=1)

        class VerifyAnyRequiredKeyAvailableInWorkingMemory(BaseModel):
            analysis: str = Field(
                description=
                "Chain-of-thought analysis of working memory and required parameters"
            )
            required_keys: Union[List[RequiredKeyValue], None] = Field(
                description=
                "List of memory keys that can be used for parameters")

        required_key_function = [{
            "name":
            "verify_any_required_key_available_in_working_memory",
            "description":
            "Analyze working memory and identify relevant keys for required parameters",
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
        system_message = """
        You are an expert at analyzing working memory and matching it with required function parameters.
        [CRITICAL RULE REGARDING MEMORY]
        - You have a dictionary called `working memory`, containing keys like:
        "swarm_0_function_name_0", "swarm_1_function_name_0", etc.
        - If you need to pass a large data structure (list, object) with more than 5 items,
        YOU MUST reference the memory key if it's already in the working memory.
        - If you inline large data that's already present in working memory, that is an error.
        - ALWAYS scan working memory for a matching or similar data structure before inlining.

        [CONSEQUENCE OF VIOLATION]
        - If you inline large data that exists in memory, your response will be rejected.
        - This is to ensure token efficiency and to avoid repeating large data.

        Your task is to:
        1. First, analyze the working memory contents and understand what type of data is available
        2. Then, examine the required parameters for the current agent's functions
        3. Use chain-of-thought reasoning to identify which memory keys could be useful for which parameters
        4. Consider:
        - Content type matching (lists, strings, query results, etc.)
        - Semantic relevance (is the content related to what the parameter needs?)
        - Data structure similarity (especially for complex objects/lists)
        - Parameter requirements vs memory content

        Here are some examples:

        Example 1 - Text Generation:
        Working Memory: {
            "swarm_0_function_extract_keywords_0": ["AI", "machine learning", "neural networks"],
            "swarm_1_function_get_context_0": "A comprehensive guide to artificial intelligence"
        }
        Analysis: "Working memory contains preprocessed keywords and context. For a text generation function requiring both keywords and context, we have exact matches available."
        Required Keys: [
            {
                "parameter_key": "keywords",
                "value_key": "swarm_0_function_extract_keywords_0",
                "reasoning": "Contains preprocessed keywords in the required list format",
                "content_type": "list",
                "similarity_score": 1.0
            }
        ]

        Example 2 - Image Analysis:
        Working Memory: {
            "swarm_0_function_process_image_0": {
                "features": ["face", "landscape"],
                "metadata": {"width": 1024, "height": 768}
            }
        }
        Analysis: "Memory contains processed image features and metadata. For an image classification function needing feature data, we have a relevant structured object."
        Required Keys: [
            {
                "parameter_key": "image_features",
                "value_key": "swarm_0_function_process_image_0",
                "reasoning": "Contains processed image features in required format",
                "content_type": "dict",
                "similarity_score": 0.9
            }
        ]

        Example 3 - Data Processing:
        Working Memory: {
            "swarm_0_function_fetch_data_0": [
                {"id": 1, "value": "test1"},
                {"id": 2, "value": "test2"},
                {"id": 3, "value": "test3"}
            ]
        }
        Analysis: "Memory contains a list of data records. For a function requiring structured data input, this matches the expected format."
        Required Keys: [
            {
                "parameter_key": "input_data",
                "value_key": "swarm_0_function_fetch_data_0",
                "reasoning": "Contains structured data in the required list format",
                "content_type": "list",
                "similarity_score": 1.0
            }
        ]"""

        user_message = f"""
        Working Memory: {json.dumps(self.working_output_memory, indent=2)}
        Current Agent Functions: {json.dumps(function_list, indent=2)}

        Following the critical rules about memory usage, analyze the working memory and identify any content that could be used for the required parameters.

        [REQUIRED ANALYSIS STEPS]
        1. First, examine all working memory keys and their content types
        2. Then, look at each required parameter in the agent functions
        3. For each parameter, check if there's matching content in memory
        4. Evaluate similarity and relevance of any potential matches
        5. Provide detailed reasoning for each match

        Remember:
        - You MUST identify any large data structures (>5 items) in memory
        - You MUST provide similarity scores for each match
        - You MUST explain your reasoning for each match
        - Failure to use available memory keys will result in rejection
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
        if len(response) == 0:
            return None
        return response[-1].get("parameters")

    async def _function_args(self,
                             model_name: str,
                             agent_name: str,
                             no_func_call: bool = False,
                             max_rec_depth: int = 3,
                             curr_depth: int = 0,
                             llm_args: Optional[Dict] = {}):
        """Get and validate function arguments for an agent's function call.

        This method handles the process of getting function arguments from the LLM,
        validating them, and ensuring they're properly formatted with memory references.

        Args:
            model_name (str): Name of the LLM model to use
            agent_name (str): Name of the agent whose function to call
            no_func_call (bool, optional): Flag for retry without function call. Defaults to False.
            max_rec_depth (int, optional): Maximum recursion depth. Defaults to 3.
            curr_depth (int, optional): Current recursion depth. Defaults to 0.
            llm_args (Optional[Dict], optional): Additional LLM parameters. Defaults to {}.

        Returns:
            Dict: Validated function arguments including:
                - name: Name of the function to call
                - parameters: Processed and validated parameters

        Raises:
            Exception: If maximum recursion depth is reached

        Example:
            >>> args = await self._function_args("gpt-4", "agent1")
            >>> print(args)
            {"name": "process", "parameters": {...}}
        """
        required_keys_available_in_working_memory = await self._verify_input_available_in_working_memory(
            model_name, agent_name, llm_args)
        function = self.swarm_map[agent_name]
        function_list = list(
            map(
                lambda x: {
                    "name": x.get("name"),
                    "description": x.get("description"),
                    "parameters": x.get("parameters")
                }, function.functions))
        curr_messages = deepcopy(self.messages)
        # print(f"Agent: {agent_name}")
        # print(f"Function: {function.name}")
        # print(f'Messages pre: {json.dumps(curr_messages, indent=2)}')
        # if curr_messages[-1].get("role") == "assistant":
        #     print("Appending user message")
        #     curr_messages.append({
        #         "role": "user",
        #         "content": function.instruction
        #     })
        # else:
        #     print("Adding message to user message")
        #     curr_messages[-1][
        #         "content"] += f"""\n\nInstruction: {function.instruction}"""
        # print(f'Messages post: {json.dumps(curr_messages, indent=2)}')
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
            memory_warning = """
            [CRITICAL MEMORY USAGE REQUIREMENT]
            The following keys are available in working memory and MUST be used instead of inlining data:
            - Failure to use these keys will result in IMMEDIATE REJECTION
            - Large data structures (>5 items) MUST use memory keys
            - Check content types and structures for matches
            - Prioritize exact matches, then similar structures
            """
            curr_messages[-1][
                "content"] += f"\n\n{memory_warning}\n{json.dumps(required_keys_available_in_working_memory, indent=2)}. Failure to use these memory keys will result in rejection."
        if no_func_call:
            curr_messages[-1][
                "content"] += f"\n\nDidn't receive a function. Please return only one appropriate function."
        # print(f"Function Args: \n{json.dumps(curr_messages, indent=2)}")
        curr_messages = [{
            "role": "system",
            "content": function.instruction
        }] + curr_messages
        # print(f"Function Args: \n{json.dumps(curr_messages, indent=2)}")
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
            response["parameters"] = self._parse_function_parameters(
                response["parameters"])
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
        async for chunk in self.llm.stream_complete(final_messages,
                                                    model_name):
            yield chunk

    def _log_error(self, message: str):
        """Log error messages using the console."""
        self.console.print(f"[red]ERROR: {message}[/red]")

    async def _process_swarm_functions(self, swarm_name: str, model_name: str,
                                       llm_args: Dict) -> AsyncGenerator:
        """Process functions for a selected swarm agent asynchronously.

        This method handles the complete lifecycle of function processing including:
        - Getting function arguments from the LLM
        - Executing the function
        - Updating the working memory
        - Yielding progress updates

        Args:
            swarm_name (str): Name of the selected swarm agent
            model_name (str): Name of the LLM model to use
            llm_args (Dict): Additional arguments for the LLM

        Yields:
            Dict: Progress updates including:
                - function_args: The arguments for the function call
                - function_result: The result of the function execution
                - error: Any error that occurred during processing

        Example:
            >>> async for result in self._process_swarm_functions("agent1", "gpt-4", {}):
            ...     print(result)
            {"function_args": {...}}
            {"function_result": {...}}
        """
        func_op = {"role": "assistant", "content": ""}
        # if self.messages[-1].get("role") != "assistant":
        #     self.messages.append(func_op)
        final_result = ""

        try:
            function_args = await self._function_args(model_name,
                                                      swarm_name,
                                                      curr_depth=0,
                                                      llm_args=llm_args)
            # print(f'Function Args: {json.dumps(function_args, indent=2)}')
            yield {"function_args": function_args}

            self.selected_agents[-1].setdefault("functions", []).append({
                "name":
                function_args.get("name"),
                "parameters":
                function_args.get("parameters")
            })

            async for chunk in self._execute_swarm_function(
                    swarm_name, function_args):
                if isinstance(chunk, dict) and ("final_result" in chunk
                                                or "error" in chunk):
                    # Store the final result
                    final_result = chunk.get("final_result") or chunk.get(
                        "error")
                else:
                    # Yield intermediate chunks for streaming
                    yield chunk
                    # yield {"streaming_chunk": chunk}
            # print(f'Final Result: {final_result}')
            # Store the final result
            self.selected_agents[-1]["functions"][-1]["result"] = final_result
            yield {"function_result": final_result}

            func_op["content"] += f"""Execution completed for:
            - Agent: {swarm_name}
            - Function: {function_args.get('name')}
            - Args: {function_args.get('parameters')}
            - Result of Execution: {final_result}\n"""
            #             self.messages[-1]["content"] += f"""Selected Agent: {swarm_name}
            # Function: {function_args.get('name')}
            # Args: {function_args.get('parameters')}
            # Result: {final_result}\n"""

            # self.messages[-1]["content"] += f"\n\n{func_op['content']}"
            if self.messages[-1].get("role") == "assistant":
                self.messages[-1]["content"] += f"\n\n{func_op['content']}"
            else:
                self.messages.append(func_op)
                # self.messages[-1]["content"] += f"""\n\n{func_op['content']}"""
            self._update_working_output_memory()

        except Exception as e:
            self._log_error(f"Function processing failed: {str(e)}")
            yield {"error": str(e)}

    async def _execute_swarm_function(self, swarm_name: str,
                                      function_args: Dict) -> AsyncGenerator:
        """Execute a single swarm function with proper async/sync handling.
        
        This method handles different types of function returns:
        - Async generators (especially for streaming responses)
        - Regular async functions
        - Synchronous functions
        - Iterables
        
        Args:
            swarm_name (str): Name of the swarm agent containing the function
            function_args (Dict): Arguments for the function including:
                - name: Name of the function to execute
                - parameters: Parameters to pass to the function
        
        Yields:
            Union[str, Dict]: Either:
                - String chunks for real-time processing
                - Dict with final result for completion
        
        Example:
            >>> async for chunk in self._execute_swarm_function("agent1", {...}):
            ...     if isinstance(chunk, dict) and "final_result" in chunk:
            ...         final_result = chunk["final_result"]
            ...     else:
            ...         print(chunk, end="")  # Stream output
        """
        agent = self.swarm_map[swarm_name]
        func = next((f["_callable"] for f in agent.functions
                     if f.get("name") == function_args.get("name")), None)

        if not func or not callable(func):
            raise ValueError(
                f"Invalid function {function_args.get('name')} for {swarm_name}"
            )

        params = function_args.get("parameters", {})

        try:
            if asyncio.iscoroutinefunction(func):
                # Handle async functions
                result = await func(**params)
            else:
                # Handle sync functions
                result = func(**params)

            # Handle different return types
            if isinstance(result, AsyncGenerator):
                # For async generators (like streaming responses)
                collected_result = ""
                async for chunk in result:
                    if isinstance(chunk, str):
                        collected_result += chunk
                        # Re-yield the chunk for real-time processing
                        yield chunk
                    else:
                        # For non-string yields, collect in a list
                        if not isinstance(collected_result, list):
                            collected_result = []
                        collected_result.append(chunk)
                # Yield final result as a dict
                yield {"final_result": collected_result}

            elif hasattr(result, '__iter__') and not isinstance(
                    result, (str, bytes, dict)):
                # For other iterables (not strings/dicts)
                yield {"final_result": list(result)}

            else:
                # For direct returns (including dicts, strings, etc.)
                yield {"final_result": result}

        except Exception as e:
            self._log_error(f"Function execution failed: {str(e)}")
            yield {"error": str(e)}

    async def _handle_completion(self, done: Dict, console: Console,
                                 write_end_result: bool) -> AsyncGenerator:
        """Handle the completion of the swarm execution and generate final answer.

        Args:
            done (Dict): Completion parameters including:
                - model_name: Name of the model to use for final answer
                - content: Content for final answer generation
            console (Console): Rich console instance for output
            write_end_result (bool): Whether to generate and yield final answer

        Yields:
            Dict: Final answer information
            str: Generated answer chunks if write_end_result is True

        Example:
            >>> async for result in self._handle_completion(done_dict, console, True):
            ...     print(result)
            {"final_answer": {...}}
            "Generated answer chunk 1"
            "Generated answer chunk 2"
        """
        console.print(f"[bold green]✨ Finalizing answer...[/bold green]")
        yield {"final_answer": done}

        if write_end_result:
            async for chunk in self._return_final_answer(
                    done.get("model_name", "default"), done):
                if isinstance(chunk, dict):
                    console.print(f"[cyan]{chunk}[/cyan]")
                elif isinstance(chunk, str):
                    console.print(f"[cyan]{chunk}[/cyan]", end="")
                yield chunk

    async def swarmloka(self,
                        model_name: str,
                        messages: List[Dict],
                        write_end_result: bool = True,
                        llm_args: Optional[Dict] = {}) -> AsyncGenerator:
        """Orchestrate the swarm execution with improved error handling and flow control.

        This is the main method that coordinates the entire swarm execution process,
        including agent selection, function execution, and result generation.

        Args:
            model_name (str): Name of the LLM model to use
            messages (List[Dict]): List of message dictionaries for context
            write_end_result (bool, optional): Whether to generate final answer. Defaults to True.
            llm_args (Optional[Dict], optional): Additional arguments for LLM. Defaults to {}.

        Yields:
            Dict: Various progress updates including:
                - status: Current status of execution
                - desired_swarm: Selected swarm information
                - error: Any errors that occurred
                - final_answer: Final generated answer

        Example:
            >>> async for result in self.swarmloka("gpt-4", messages):
            ...     print(result)
            {"desired_swarm": [...]}
            {"function_args": {...}}
            {"function_result": {...}}
            {"final_answer": {...}}

        Notes:
            - The method will continue until either:
                1. The maximum iterations are reached
                2. An "end" function is called
                3. An error occurs
            - Each iteration involves:
                1. Getting the next action from the orchestrator
                2. Processing the selected agent's functions
                3. Updating the working memory
                4. Yielding progress updates
        """
        self.messages = messages
        self.current_iteration = 0

        while self.current_iteration <= self.max_iterations:
            if self.current_iteration >= self.max_iterations:
                yield {"status": "max_iterations_reached"}
                break

            with Status("[bold blue]🤖 Orchestrator thinking...",
                        console=self.console) as status:
                try:
                    desired_swarm = await self._iterate(model_name, llm_args)
                except Exception as e:
                    self._log_error(f"Orchestrator iteration failed: {str(e)}")
                    break

                yield {"desired_swarm": desired_swarm}

                try:
                    selected_agent = desired_swarm[0].get("name")
                    if selected_agent == "end":
                        done = desired_swarm[0].get("parameters")
                        async for chunk in self._handle_completion(
                                done, self.console, write_end_result):
                            yield chunk
                        break

                    # self._update_status(status, f"Selected: {selected_agent}",
                    #                     console)
                    status.update(f"[bold cyan]{selected_agent}[/bold cyan]")
                    self.selected_agents.append({"agent": selected_agent})

                    async for result in self._process_swarm_functions(
                            selected_agent, model_name, llm_args):
                        yield result
                    # print(selected_agent)
                    # print(self.swarm_map[selected_agent].model_dump())
                    # print(self.swarm_map[selected_agent].exit_here)
                    if self.swarm_map[selected_agent].exit_here:
                        status.stop()
                        # async for chunk in self._handle_completion(
                        #         done, console, write_end_result):
                        #     yield chunk
                        break

                except Exception as e:
                    self._log_error(f"Swarm processing failed: {str(e)}")
                    self._log_error(f"Traceback: {traceback.format_exc()}")
                    yield {"error": f"Processing error: {str(e)}"}
                    break

                self.current_iteration += 1
