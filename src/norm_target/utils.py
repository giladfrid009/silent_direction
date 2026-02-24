from src.utils.logging import create_logger


logger = create_logger(__name__)


def score_target_norm(proj_norm: float, target_norm: float) -> float:
    return 1 - abs(proj_norm - target_norm) / target_norm


