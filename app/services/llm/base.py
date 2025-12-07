from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseLLMClient(Protocol):
    async def generate_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate structured data from a prompt."""
        ...
