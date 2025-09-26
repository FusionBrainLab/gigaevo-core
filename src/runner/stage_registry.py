"""Minimal stage registry for DAG builder GUI.

Automatically extracts input information from stage classes using decorators.
"""

from typing import Dict, List, Type, Any, Optional
from dataclasses import dataclass
from src.programs.stages.base import Stage


@dataclass
class StageInfo:
    """Information about a stage for the GUI."""
    name: str
    description: str
    class_name: str
    import_path: str
    mandatory_inputs: List[str]
    optional_inputs: List[str]


class StageRegistry:
    """Simple registry of stage classes with auto-extraction of inputs."""
    
    _stages: Dict[str, StageInfo] = {}
    
    @classmethod
    def register(
        cls,
        description: str = "",
        import_path: Optional[str] = None
    ):
        """Decorator to register a stage class.
        
        Args:
            description: Description for the GUI
            import_path: Import path (auto-detected if None)
        """
        def decorator(stage_class: Type[Stage]) -> Type[Stage]:
            # Use class name as the registry key
            class_name = stage_class.__name__
            
            # Auto-extract inputs from static methods
            mandatory_inputs = []
            optional_inputs = []
            mandatory_inputs = stage_class.mandatory_inputs()
            optional_inputs = stage_class.optional_inputs()
            
            # Auto-detect import path if not provided
            final_import_path = import_path
            if final_import_path is None:
                final_import_path = f"{stage_class.__module__}.{class_name}"
            
            cls._stages[class_name] = StageInfo(
                name=class_name,
                description=description,
                class_name=class_name,
                import_path=final_import_path,
                mandatory_inputs=mandatory_inputs,
                optional_inputs=optional_inputs
            )
            
            return stage_class
        return decorator
    
    @classmethod
    def get_all_stages(cls) -> Dict[str, StageInfo]:
        """Get all registered stages."""
        return cls._stages.copy()
    
    @classmethod
    def get_stage(cls, name: str) -> Optional[StageInfo]:
        """Get a specific stage by name."""
        return cls._stages.get(name)
    
    @classmethod
    def clear(cls):
        """Clear all registered stages (for testing)."""
        cls._stages.clear()
