from typing import List, TYPE_CHECKING
from loguru import logger
from src.programs.program import Program
from .models import QualityDiversityMetrics
from src.evolution.strategies.utils import extract_fitness_values

if TYPE_CHECKING:
    from .island import IslandConfig

def compute_qd_metrics_for_island(elites: List[Program], config: "IslandConfig") -> QualityDiversityMetrics:
    if not elites:
        return QualityDiversityMetrics(
            qd_score=0.0,
            coverage=0.0,
            maximum_fitness=float('-inf'),
            average_fitness=0.0,
            filled_cells=0,
            total_cells=config.behavior_space.total_cells
        )

    # Use fitness keys from archive selector instead of hardcoding "fitness"
    fitness_keys = config.archive_selector.fitness_keys
    fitness_key_higher_is_better = config.archive_selector.fitness_key_higher_is_better

    # Calculate fitness values using the archive selector's scoring method
    fitness_values = []
    for elite in elites:
        try:
            if hasattr(config.archive_selector, 'score'):
                # Use the archive selector's scoring method if available
                fitness_values.append(config.archive_selector.score(elite))
            else:
                # Fallback to sum of fitness keys
                values = extract_fitness_values(elite, fitness_keys, fitness_key_higher_is_better)
                fitness_values.append(sum(values))
        except Exception as e:
            logger.warning(f"Error calculating fitness for elite {elite.id}: {e}")
            fitness_values.append(0.0)

    max_fitness = max(fitness_values) if fitness_values else 0.0
    avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
    qd_score = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
    filled_cells = len(elites)
    total_cells = config.behavior_space.total_cells
    coverage = filled_cells / total_cells if total_cells > 0 else 0.0

    metrics = QualityDiversityMetrics(
        qd_score=qd_score,
        coverage=coverage,
        maximum_fitness=max_fitness,
        average_fitness=avg_fitness,
        filled_cells=filled_cells,
        total_cells=total_cells
    )
    return metrics 