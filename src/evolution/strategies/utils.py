from src.programs.program import Program
from typing import List
import math

def extract_fitness_values(program: Program, fitness_keys: List[str], fitness_key_higher_is_better: dict[str, bool]) -> List[float]:
        """Extract fitness values with comprehensive error handling."""
        # by default, all fitness keys are assumed to be higher is better
        if not isinstance(program.metrics, dict):
            raise ValueError("Program metrics must be a dictionary")
        
        assert set(fitness_keys) == set(fitness_key_higher_is_better.keys()), "All fitness keys must be present in the fitness_key_higher_is_better dict"
            
        values = []
        for key in fitness_keys:
            if key not in program.metrics:
                raise KeyError(f"Missing fitness key '{key}' in program metrics")
            
            raw_value = program.metrics[key]
            
            # Check for None or invalid types
            if raw_value is None:
                raise ValueError(f"Invalid fitness value for key '{key}': None")
            
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid fitness value for key '{key}': {raw_value} (cannot convert to float)")
            
            # Check for NaN
            if math.isnan(value):
                raise ValueError(f"Fitness value for key '{key}' is NaN")
            
            # Check for infinity
            if math.isinf(value):
                raise ValueError(f"Fitness value for key '{key}' is infinite")
            
            values.append(value if fitness_key_higher_is_better[key] else -value)
        return values

def dominates(p: List[float], q: List[float]) -> bool:
    """Returns True if p Pareto-dominates q (i.e., p is ≥ in all and > in at least one)."""
    return all(p_i >= q_i for p_i, q_i in zip(p, q)) and any(p_i > q_i for p_i, q_i in zip(p, q))