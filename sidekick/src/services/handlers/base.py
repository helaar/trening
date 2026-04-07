from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskHandler(Protocol):
    """Protocol for task-specific execution handlers."""

    async def execute(
        self,
        task_id: str,
        athlete_id: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]: ...
