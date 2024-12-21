import json
from openai import AsyncOpenAI
from typing import Dict, List, Optional, AsyncGenerator
import re
from .base import BaseLLM
from .prompts import LOCAL_SINGLE_FUNCTION_CALL_PROMPT, LOCAL_MULTI_FUNCTION_CALL_PROMPT


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
        # print(f"RESPONSE: {response}")
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
        # print(f"FUNCTION CALL: {function_call}s")
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
            if multi_call:
                function_call_prompt = LOCAL_MULTI_FUNCTION_CALL_PROMPT.PROMPT.format(
                    functions=functions)
            else:
                function_call_prompt = LOCAL_SINGLE_FUNCTION_CALL_PROMPT.PROMPT.format(
                    functions=functions)
                if function_call != "auto":
                    last_message = messages[-1]
                    content = last_message.get("content", "")
                    content = content + f" Use function {{'function_call': '{function_call}'}}"
                    # print(f"NOT AUTO FUNCTION CALLCONTENT:\n {content}")
                    messages[-1]["content"] = content
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
