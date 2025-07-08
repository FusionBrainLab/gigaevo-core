"""Constants for the metaevolve pipeline."""

# Regex patterns for dangerous code constructs
DANGEROUS_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+glob\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+urllib\b",
    r"\bimport\s+requests\b",
    r"\bimport\s+pickle\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(",
    r"\bfile\s*\(",
    r"\binput\s*\(",
    r"\braw_input\s*\(",
]

# Default timeouts (in seconds)
DEFAULT_STAGE_TIMEOUT = 30.0

# Common stage names - these are part of the API
STAGE_VALIDATE_CODE = "validate_code"
STAGE_RUN_CODE = "run_code"
STAGE_RUN_VALIDATION = "run_validation"
STAGE_UPDATE_METRICS = "update_metrics"

# Stage execution order
DEFAULT_STAGE_ORDER = [
    STAGE_VALIDATE_CODE,
    STAGE_RUN_CODE,
    STAGE_RUN_VALIDATION,
    STAGE_UPDATE_METRICS,
]
