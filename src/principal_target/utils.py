from src.utils.logging import create_logger


logger = create_logger(__name__)


def score_principal_target(proj_var: float, target_var: float) -> float:
    return 1 - abs(proj_var - target_var) / target_var

