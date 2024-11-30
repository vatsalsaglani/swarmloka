import json
from openai import AsyncOpenAI
from typing import Dict, List, Optional, AsyncGenerator
import re
from .base import BaseLLM


class LocalLLM(BaseLLM):

    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def complete(self, messages: List[Dict], model_name: str,
                       **kwargs) -> str:
        response = await self.client.chat.completions.create(model=model_name,
                                                             messages=messages,
                                                             **kwargs)
        return response.choices[0].message.content

    async def stream_complete(self, messages: List[Dict], model_name: str,
                              **kwargs):
        try:
            response = await self.client.chat.completions.create(
                model=model_name, messages=messages, stream=True, **kwargs)
            async for chunk in response:
                if chunk.choices[0].delta and chunk.choices[
                        0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise e

    def _extract_function_calls(self, response: str) -> List[Dict]:
        pattern = r'<functioncall>\s*(.*?)\s*</functioncall>'
        function_calls = re.findall(pattern, response)

        parsed_calls = []
        for call in function_calls:
            try:
                parsed_calls.append(json.loads(call))
            except json.JSONDecodeError:
                continue
        return parsed_calls

    def _validate_parameters(self, provided_params: Dict,
                             expected_params: Dict) -> List[str]:
        """Validate parameters against schema and return list of validation errors."""
        validation_errors = []

        required_params = expected_params.get('required', [])
        properties = expected_params.get('properties', {})

        missing_params = [
            p for p in required_params if p not in provided_params
        ]
        if missing_params:
            validation_errors.append(
                f"Missing required parameters: {missing_params}")

        for param_name, param_value in provided_params.items():
            if param_name not in properties:
                validation_errors.append(f"Unexpected parameter: {param_name}")
                continue

            expected_type = properties[param_name].get('type')
            if expected_type == 'array' and not isinstance(param_value, list):
                validation_errors.append(
                    f"Parameter {param_name} should be an array")
            elif expected_type == 'integer' and not isinstance(
                    param_value, int):
                validation_errors.append(
                    f"Parameter {param_name} should be an integer")
            elif expected_type == 'string' and not isinstance(
                    param_value, str):
                validation_errors.append(
                    f"Parameter {param_name} should be a string")
            elif expected_type == 'number' and not isinstance(
                    param_value, (int, float)):
                validation_errors.append(
                    f"Parameter {param_name} should be a number")
            elif expected_type == 'boolean' and not isinstance(
                    param_value, bool):
                validation_errors.append(
                    f"Parameter {param_name} should be a boolean")

        return validation_errors

    async def _recursive_function_call(self,
                                       messages: List[Dict],
                                       model_name: str,
                                       functions: List[Dict],
                                       multi_call: bool = False,
                                       validate_params: bool = True,
                                       depth: int = 0,
                                       max_depth: int = 3,
                                       **kwargs):
        if depth >= max_depth:
            return []

        response = await self.complete(messages, model_name, **kwargs)
        extracted_functions = self._extract_function_calls(response)

        if not extracted_functions:
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. You did not select any function. Please select one appropriate function."
            return await self._recursive_function_call(messages, model_name,
                                                       functions, multi_call,
                                                       validate_params,
                                                       depth + 1, max_depth,
                                                       **kwargs)
        if len(extracted_functions) > 1 and not multi_call:
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. You selected multiple functions. Please select only one function."
            return await self._recursive_function_call(messages, model_name,
                                                       functions, multi_call,
                                                       validate_params,
                                                       depth + 1, max_depth,
                                                       **kwargs)
        function_call = extracted_functions[0]
        function_name = function_call.get('name')
        target_function = next(
            (f for f in functions if f['name'] == function_name), None)

        if not target_function:
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. You selected a function that does not exist. Please select a valid function."
            return await self._recursive_function_call(messages, model_name,
                                                       functions, multi_call,
                                                       validate_params,
                                                       depth + 1, max_depth,
                                                       **kwargs)

        validation_errors = None
        if validate_params:
            validation_errors = self._validate_parameters(
                function_call.get("parameters", {}),
                target_function.get("parameters", {}))

        if validation_errors:
            error_message = "\n".join(validation_errors)
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. The parameters are not valid: {error_message}. Please select the most appropriate function with valid parameters."
            return await self._recursive_function_call(messages, model_name,
                                                       functions, multi_call,
                                                       validate_params,
                                                       depth + 1, max_depth,
                                                       **kwargs)

        if isinstance(function_call, dict):
            function_call = [function_call]
        return function_call

    async def function_call(self,
                            messages: List[Dict],
                            model_name: str,
                            functions: List[Dict],
                            function_call: str = "auto",
                            multi_call: Optional[bool] = None,
                            validate_params: Optional[bool] = None,
                            **kwargs):

        try:
            # TODO: logic for multi-call and function_call "auto" or selected
            function_call_prompt = """You are a helpful AI assistant with access to a set of functions. Your task is to assist users by utilizing these functions to respond to their requests. Here are your instructions:

            1. Available Functions:
            <functions>
            {functions}
            </functions>

            2. Using Functions:
            - You MUST provide exactly ONE function call per response
            - Your response MUST be enclosed in <functioncall> tags
            - Parameters MUST match the function's schema exactly
            - Do not provide any additional text or explanations
            - For multi-step tasks, determine which step is currently needed based on the conversation history
            - When all steps are complete, use the 'end' function

            3. Format:
            <functioncall> {{"name": "functionName", "parameters": {{"param1": "value1", "param2": "value2"}} }} </functioncall>

            4. Few-Shot Examples:

            User: "First check the weather in London, then book a taxi"
            Assistant: <functioncall> {{"name": "getWeather", "parameters": {{"city": "London"}} }} </functioncall>

            System: Weather in London is 15°C and sunny
            User: "First check the weather in London, then book a taxi"
            Assistant: <functioncall> {{"name": "bookTaxi", "parameters": {{"pickup": "London"}} }} </functioncall>

            User: "Convert 100F to Celsius and then send it to admin@example.com"
            Assistant: <functioncall> {{"name": "convertTemperature", "parameters": {{"value": 100, "from": "F", "to": "C"}} }} </functioncall>

            System: Temperature converted to 37.8C
            User: "Convert 100F to Celsius and then send it to admin@example.com"
            Assistant: <functioncall> {{"name": "sendEmail", "parameters": {{"to": "admin@example.com", "body": "Temperature is 37.8C"}} }} </functioncall>

            5. Important:
            - Examine the conversation history to determine which step needs to be executed next
            - Return exactly ONE function call that matches the current step
            - If all steps are complete, use the 'end' function with appropriate parameters
            - Never include multiple function calls in one response
            - Return the function call in <functioncall> tags as shown in the examples above

            Remember: Only ONE function call is allowed per response. Determine the current step from the conversation history and return only one function call.
            """.format(functions=functions)
            _messages = [{
                "role": "system",
                "content": function_call_prompt
            }] + messages
            depth = 0
            max_depth = 5
            return await self._recursive_function_call(_messages, model_name,
                                                       functions, multi_call,
                                                       validate_params, depth,
                                                       max_depth, **kwargs)

        except Exception as e:
            raise e
