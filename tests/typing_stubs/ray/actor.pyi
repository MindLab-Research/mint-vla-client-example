from typing import Any

class ActorHandle:
    def __getattr__(self, name: str) -> Any: ...
