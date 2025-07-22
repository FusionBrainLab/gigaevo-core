"""
Enhanced error handling utilities for MetaEvolve system.
Provides circuit breakers, retry mechanisms, and graceful degradation.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import time
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit breaker open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Time to wait before trying again
    success_threshold: int = 3  # Consecutive successes needed to close
    timeout: float = 30.0  # Operation timeout


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute a function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (
                    time.time() - self.last_failure_time
                    < self.config.recovery_timeout
                ):
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is open"
                    )
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(
                        f"Circuit breaker {self.name} transitioning to HALF_OPEN"
                    )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(), timeout=self.config.timeout)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful execution."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(
                        f"Circuit breaker {self.name} closed after recovery"
                    )
            else:
                self.failure_count = 0

    async def _on_failure(self):
        """Handle failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(
                    f"Circuit breaker {self.name} opened after {self.failure_count} failures"
                )

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    config: RetryConfig = None,
    exception_types: tuple = (Exception,),
    operation_name: str = "operation",
) -> T:
    """Retry a function with exponential backoff and jitter."""
    config = config or RetryConfig()

    for attempt in range(config.max_attempts):
        try:
            return await func()
        except exception_types as e:
            if attempt == config.max_attempts - 1:
                logger.error(
                    f"Failed {operation_name} after {config.max_attempts} attempts: {e}"
                )
                raise

            delay = min(
                config.base_delay * (config.exponential_base**attempt),
                config.max_delay,
            )

            if config.jitter:
                import random

                delay *= (
                    0.5 + random.random() * 0.5
                )  # 50-100% of calculated delay

            logger.warning(
                f"Attempt {attempt + 1} failed for {operation_name}, retrying in {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Retry logic failed for {operation_name}")


class HealthChecker:
    """Health monitoring for system components."""

    def __init__(self):
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def check_health(
        self, component: str, health_check: Callable[[], Awaitable[bool]]
    ) -> bool:
        """Check health of a component."""
        try:
            is_healthy = await asyncio.wait_for(health_check(), timeout=10.0)
            await self._update_health_status(component, is_healthy, None)
            return is_healthy
        except Exception as e:
            await self._update_health_status(component, False, str(e))
            return False

    async def _update_health_status(
        self, component: str, is_healthy: bool, error: Optional[str]
    ):
        """Update health status for a component."""
        async with self._lock:
            self.health_status[component] = {
                "healthy": is_healthy,
                "last_check": time.time(),
                "error": error,
            }

    async def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current health status of all components."""
        async with self._lock:
            return self.health_status.copy()


# Global instances
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_health_checker = HealthChecker()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to add circuit breaker protection to async functions."""

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if name not in _circuit_breakers:
                _circuit_breakers[name] = CircuitBreaker(name, config)

            breaker = _circuit_breakers[name]
            return await breaker.call(lambda: func(*args, **kwargs))

        return wrapper

    return decorator


def resilient_operation(
    retry_config: RetryConfig = None,
    circuit_breaker_name: str = None,
    circuit_breaker_config: CircuitBreakerConfig = None,
    operation_name: str = None,
):
    """Decorator combining retry logic and circuit breaker protection."""

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or func.__name__

            async def protected_call():
                if circuit_breaker_name:
                    if circuit_breaker_name not in _circuit_breakers:
                        _circuit_breakers[circuit_breaker_name] = (
                            CircuitBreaker(
                                circuit_breaker_name, circuit_breaker_config
                            )
                        )
                    breaker = _circuit_breakers[circuit_breaker_name]
                    return await breaker.call(lambda: func(*args, **kwargs))
                else:
                    return await func(*args, **kwargs)

            return await retry_with_backoff(
                protected_call, retry_config, operation_name=op_name
            )

        return wrapper

    return decorator


async def get_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    status = {}
    for name, breaker in _circuit_breakers.items():
        status[name] = {
            "state": breaker.state.value,
            "failure_count": breaker.failure_count,
            "success_count": breaker.success_count,
            "last_failure_time": breaker.last_failure_time,
        }
    return status


async def get_system_health() -> Dict[str, Any]:
    """Get overall system health status."""
    return {
        "circuit_breakers": await get_circuit_breaker_status(),
        "component_health": await _health_checker.get_health_status(),
    }
