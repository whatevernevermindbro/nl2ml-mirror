from .private import private
from .public import public
from .types import ModuleAware


__version__ = '2.1.1'


def install() -> None:
    """Install @public and @private into builtins."""
    import builtins
    builtins.public = public            # type: ignore [attr-defined]
    builtins.private = private          # type: ignore [attr-defined]


public(
    ModuleAware=ModuleAware,
    public=public,
    private=private,
    )
