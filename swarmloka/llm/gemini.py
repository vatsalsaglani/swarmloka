from openai import AsyncOpenAI
from typing import Dict, List, Optional, AsyncGenerator
import re
from .base import BaseLLM


class GeminiLLM(BaseLLM):

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
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error streaming completion: {e}")
            yield ""

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
                                       function_call: str = "auto",
                                       multi_call: Optional[bool] = None,
                                       validate_params: Optional[bool] = None,
                                       depth: int = 0,
                                       max_depth: int = 5,
                                       **kwargs):
        if depth >= max_depth:
            return []

        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=functions,
            tool_choice=function_call,
            **kwargs)
        if function_call.get("function").get("name") == "auto":
            print(f"FUNCTION CALL AUTO")
            function_response = response.choices[0].message.tool_calls[
                0].function
        else:
            print(f"FUNCTION CALL NOT AUTO")
            function_response = response.choices[0].message.tool_calls[
                0].function
            if function_response.name != function_call.get("function").get(
                    "name"):
                print(
                    f"FUNCTION CALL NOT MATCHING: {function_response.name} != {function_call.get('function').get('name')}"
                )
                messages[-1][
                    "content"] += f"\n\nThe following is your output: {response}. You were asked to call {function_call.get('function').get('name')} but instead called {function_response.name}. Please call {function_call.get('function').get('name')} instead."
                return await self._recursive_function_call(
                    messages, model_name, functions, function_call, multi_call,
                    validate_params, depth + 1, max_depth, **kwargs)

        validation_errors = None
        if validate_params:
            validation_errors = self._validate_parameters(
                function_response.parameters,
                function_call.get("parameters", {}))

        if validation_errors:
            print(f"VALIDATION ERRORS: {validation_errors}")
            error_message = "\n".join(validation_errors)
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. The parameters are not valid: {error_message}. Please select the most appropriate function with valid parameters."
            return await self._recursive_function_call(
                messages, model_name, functions, function_call, multi_call,
                validate_params, depth + 1, max_depth, **kwargs)

        function_response = [{
            "name": function_response.name,
            "parameters": function_response.parameters
        }]
        return function_response

    async def function_call(self,
                            messages: List[Dict],
                            model_name: str,
                            functions: List[Dict],
                            function_call: str = "auto",
                            multi_call: Optional[bool] = None,
                            validate_params: Optional[bool] = None,
                            **kwargs):
        functions = list(
            map(lambda x: {
                "type": "function",
                "function": x
            }, functions))
        function_call = {
            "type": "function",
            "function": {
                "name": function_call
            }
        }
        return await self._recursive_function_call(messages,
                                                   model_name,
                                                   functions,
                                                   function_call,
                                                   multi_call,
                                                   validate_params,
                                                   depth=0,
                                                   max_depth=5,
                                                   **kwargs)
