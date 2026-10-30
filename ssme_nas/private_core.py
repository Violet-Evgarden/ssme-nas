"""Explicit missing implementation, not an intentionally broken real algorithm."""


class CoreNotReleasedError(RuntimeError):
    """The scientific implementation is not distributed in this preview."""


def require_private_core():
    raise CoreNotReleasedError(
        "Unpublished core modules are not distributed. This checkout cannot run "
        "SSME-NAS or reproduce its experiments. See docs/RELEASE_STATUS.md."
    )


def compile_memory(*args, **kwargs):
    require_private_core()


def propose_mutation(*args, **kwargs):
    require_private_core()
