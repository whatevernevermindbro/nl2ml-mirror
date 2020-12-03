try:
    from typing import Protocol
except ImportError:                                 # pragma: nocover
    # Python < 3.8
    from typing_extensions import Protocol  # type: ignore


class ModuleAware(Protocol):
    __module__: str
    __name__: str
