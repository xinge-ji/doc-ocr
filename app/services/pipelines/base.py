from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BasePipeline(Protocol):
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the pipeline."""
        ...
