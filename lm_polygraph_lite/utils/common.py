import logging
import importlib.util

log = logging.getLogger("lm_polygraph_lite")


def polygraph_module_init(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "__init__":
            log.info(f"Initializing {args[0].__class__.__name__}")
        func(*args, **kwargs)

    return wrapper


def load_external_module(path_to_file: str):
    """Load external module from file and return it."""

    spec = importlib.util.spec_from_file_location("external_module", path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
