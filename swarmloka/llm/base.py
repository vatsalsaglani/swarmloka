from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseLLM(ABC):

    @abstractmethod
    async def complete(self, messages: List[Dict], model_name: str,
                       **kwargs) -> str:
        pass

    @abstractmethod
    async def stream_complete(self, messages: List[Dict], model_name: str,
                              **kwargs):
        pass

    @abstractmethod
    async def function_call(self,
                            messages: List[Dict],
                            model_name: str,
                            functions: List[Dict],
                            function_call: str = "auto",
                            multi_call: Optional[bool] = None,
                            validate_params: Optional[bool] = None,
                            **kwargs) -> Dict:
        pass
