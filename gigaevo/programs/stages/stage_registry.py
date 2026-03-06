from dataclasses import dataclass
import typing
from typing import Type, get_args, get_origin

from gigaevo.programs.stages.base import Stage


@dataclass
class StageInfo:
    """Information about a stage for the GUI."""

    name: str
    description: str
    class_name: str
    import_path: str
    mandatory_inputs: list[str]
    optional_inputs: list[str]
    input_types: dict[str, str]  # Mapping of input name to type annotation string
    output_fields: list[str]  # Fields from OutputModel
    output_model_name: str  # Name of the OutputModel class


class StageRegistry:
    """Simple registry of stage classes with auto-extraction of inputs."""

    _stages: dict[str, StageInfo] = {}

    @classmethod
    def register(cls, description: str = "", import_path: str | None = None):
        """Decorator to register a stage class.

        Args:
            description: Description for the GUI
            import_path: Import path (auto-detected if None)
        """

        def decorator(stage_class: Type[Stage]) -> Type[Stage]:
            # Use class name as the registry key
            class_name = stage_class.__name__

            # Extract input names (already validated by Stage.__init_subclass__)
            mandatory_inputs = stage_class._required_names
            optional_inputs = stage_class._optional_names

            # Extract input types from Pydantic model

            def format_type_name(annotation) -> str:
                """Format a type annotation into a readable string."""
                if annotation is type(None):
                    return "None"

                # Check for generic types FIRST before checking __name__
                origin = get_origin(annotation)
                args = get_args(annotation)

                # Handle Union types (including Optional)
                if origin is typing.Union:
                    # Check if it's Optional (Union[X, None])
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    if len(non_none_args) == 1 and type(None) in args:
                        # It's Optional[X]
                        inner_type = format_type_name(non_none_args[0])
                        return f"Optional[{inner_type}]"
                    else:
                        # It's Union[X, Y, ...]
                        formatted_args = [format_type_name(arg) for arg in args]
                        return f"Union[{', '.join(formatted_args)}]"

                # Handle other generic types (List, Dict, etc.)
                if origin is not None:
                    origin_name = getattr(origin, "__name__", str(origin))
                    if args:
                        formatted_args = [format_type_name(arg) for arg in args]
                        return f"{origin_name}[{', '.join(formatted_args)}]"
                    return origin_name

                # Simple types (no origin, not generic)
                if hasattr(annotation, "__name__"):
                    return annotation.__name__

                # Fallback to string representation
                return str(annotation).replace("typing.", "")

            input_types = {}
            for field_name, field_info in stage_class.InputsModel.model_fields.items():
                input_types[field_name] = format_type_name(field_info.annotation)

            # Extract output fields from Pydantic model
            output_model_name = stage_class.OutputModel.__name__
            output_fields = list(stage_class.OutputModel.model_fields.keys())

            # Auto-detect import path if not provided
            final_import_path = import_path or f"{stage_class.__module__}.{class_name}"

            cls._stages[class_name] = StageInfo(
                name=class_name,
                description=description,
                class_name=class_name,
                import_path=final_import_path,
                mandatory_inputs=mandatory_inputs,
                optional_inputs=optional_inputs,
                input_types=input_types,
                output_fields=output_fields,
                output_model_name=output_model_name,
            )

            return stage_class

        return decorator

    @classmethod
    def get_all_stages(cls) -> dict[str, StageInfo]:
        """Get all registered stages."""
        return cls._stages.copy()

    @classmethod
    def get_stage(cls, name: str) -> StageInfo | None:
        """Get a specific stage by name."""
        return cls._stages.get(name)

    @classmethod
    def clear(cls):
        """Clear all registered stages (for testing)."""
        cls._stages.clear()
