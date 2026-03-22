# Lazy imports to avoid pulling torch when only database is needed
from .database import EvaluationDatabase
from .candidates import generate_candidates
from .catalog import DesignCatalog, CatalogEntry


def run_surrogate_assisted(*args, **kwargs):
    from .problem import run_surrogate_assisted as _run
    return _run(*args, **kwargs)


def run_nsga2_assisted(*args, **kwargs):
    from .problem import run_nsga2_assisted as _run
    return _run(*args, **kwargs)
