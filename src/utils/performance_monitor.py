"""
Comprehensive performance monitoring for MetaEvolve system.
Tracks metrics, health status, and optimization effectiveness.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timezone

from loguru import logger


@dataclass
class PerformanceMetrics:
    """Core performance metrics for system monitoring."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    redis_operations_per_sec: float
    dag_completion_rate: float
    evolution_iterations_per_sec: float
    active_connections: int
    cache_hit_rate: float
    error_rate: float


@dataclass
class SystemHealth:
    """Overall system health status."""
    timestamp: float
    status: str  # "healthy", "degraded", "critical"
    redis_status: str
    llm_status: str
    dag_processor_status: str
    evolution_engine_status: str
    total_errors_last_hour: int
    performance_score: float  # 0-100


class MetricsCollector:
    """Collects and aggregates system performance metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.cache_stats = {"hits": 0, "misses": 0}
        self.start_time = time.time()
        self._lock = asyncio.Lock()
    
    async def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record an operation with its duration and success status."""
        async with self._lock:
            self.operation_counts[operation] += 1
            self.response_times[operation].append(duration)
            
            if not success:
                self.error_counts[operation] += 1
    
    async def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        async with self._lock:
            if hit:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
    
    async def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # Calculate rates
        uptime = time.time() - self.start_time
        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.error_counts.values())
        
        redis_ops_per_sec = self.operation_counts.get("redis", 0) / max(uptime, 1)
        dag_completion_rate = self.operation_counts.get("dag_complete", 0) / max(uptime, 1)
        evolution_rate = self.operation_counts.get("evolution_step", 0) / max(uptime, 1)
        
        # Cache hit rate
        total_cache_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / max(total_cache_ops, 1)
        
        # Error rate
        error_rate = total_errors / max(total_operations, 1)
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory.percent,
            redis_operations_per_sec=redis_ops_per_sec,
            dag_completion_rate=dag_completion_rate,
            evolution_iterations_per_sec=evolution_rate,
            active_connections=self.operation_counts.get("active_connections", 0),
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    async def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get detailed statistics for a specific operation."""
        async with self._lock:
            times = list(self.response_times[operation])
            if not times:
                return {"count": 0, "avg_time": 0, "min_time": 0, "max_time": 0}
            
            return {
                "count": self.operation_counts[operation],
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "error_count": self.error_counts[operation],
                "error_rate": self.error_counts[operation] / max(self.operation_counts[operation], 1)
            }
    
    async def get_recent_metrics(self, seconds: int = 300) -> List[PerformanceMetrics]:
        """Get metrics from the last N seconds."""
        cutoff = time.time() - seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff]


class HealthMonitor:
    """Monitors system health and provides status assessments."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: deque = deque(maxlen=100)
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "error_rate": 0.1,
            "redis_operations_per_sec": 1.0,
            "evolution_iterations_per_sec": 0.1
        }
    
    def register_health_check(self, component: str, check_func: Callable[[], bool]):
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
    
    async def assess_system_health(self) -> SystemHealth:
        """Assess overall system health."""
        metrics = await self.metrics_collector.collect_system_metrics()
        
        # Component status checks
        redis_status = "healthy"
        llm_status = "healthy"
        dag_processor_status = "healthy"
        evolution_engine_status = "healthy"
        
        # Check thresholds
        alerts = []
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append("high_cpu")
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append("high_memory")
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append("high_error_rate")
        if metrics.redis_operations_per_sec < self.alert_thresholds["redis_operations_per_sec"]:
            redis_status = "degraded"
        if metrics.evolution_iterations_per_sec < self.alert_thresholds["evolution_iterations_per_sec"]:
            evolution_engine_status = "degraded"
        
        # Overall status
        if len(alerts) >= 3:
            overall_status = "critical"
        elif len(alerts) >= 1:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Performance score (0-100)
        performance_score = 100.0
        performance_score -= len(alerts) * 15  # -15 points per alert
        performance_score -= max(0, metrics.cpu_percent - 50) * 0.5  # CPU penalty
        performance_score -= max(0, metrics.memory_percent - 70) * 0.5  # Memory penalty
        performance_score = max(0, min(100, performance_score))
        
        # Count recent errors
        recent_metrics = await self.metrics_collector.get_recent_metrics(3600)  # Last hour
        total_errors = sum(m.error_rate * 100 for m in recent_metrics)  # Rough estimate
        
        health = SystemHealth(
            timestamp=time.time(),
            status=overall_status,
            redis_status=redis_status,
            llm_status=llm_status,
            dag_processor_status=dag_processor_status,
            evolution_engine_status=evolution_engine_status,
            total_errors_last_hour=int(total_errors),
            performance_score=performance_score
        )
        
        self.health_history.append(health)
        return health
    
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current system alerts."""
        health = await self.assess_system_health()
        metrics = await self.metrics_collector.collect_system_metrics()
        
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "level": "warning",
                "component": "system",
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                "metric": "cpu_percent",
                "value": metrics.cpu_percent,
                "threshold": self.alert_thresholds["cpu_percent"]
            })
        
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "level": "critical",
                "component": "system",
                "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                "metric": "memory_percent",
                "value": metrics.memory_percent,
                "threshold": self.alert_thresholds["memory_percent"]
            })
        
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "level": "critical",
                "component": "application",
                "message": f"High error rate: {metrics.error_rate:.2%}",
                "metric": "error_rate",
                "value": metrics.error_rate,
                "threshold": self.alert_thresholds["error_rate"]
            })
        
        return alerts


class PerformanceOptimizer:
    """Automatically applies performance optimizations based on metrics."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.optimization_history: List[Dict[str, Any]] = []
        self.auto_optimize_enabled = True
    
    async def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest optimizations based on current metrics."""
        metrics = await self.health_monitor.metrics_collector.collect_system_metrics()
        suggestions = []
        
        # Memory optimization suggestions
        if metrics.memory_percent > 80:
            suggestions.append({
                "type": "memory",
                "action": "Force garbage collection",
                "reason": f"Memory usage at {metrics.memory_percent:.1f}%",
                "auto_apply": True
            })
        
        # Redis optimization suggestions
        if metrics.redis_operations_per_sec < 0.5:
            suggestions.append({
                "type": "redis",
                "action": "Check Redis connection pool",
                "reason": f"Low Redis throughput: {metrics.redis_operations_per_sec:.2f} ops/sec",
                "auto_apply": False
            })
        
        # Evolution engine suggestions
        if metrics.evolution_iterations_per_sec < 0.1:
            suggestions.append({
                "type": "evolution",
                "action": "Reduce DAG parallelism",
                "reason": f"Evolution stalled: {metrics.evolution_iterations_per_sec:.3f} iter/sec",
                "auto_apply": False
            })
        
        return suggestions
    
    async def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply an optimization if enabled."""
        if not self.auto_optimize_enabled:
            return False
        
        try:
            if optimization["type"] == "memory" and optimization["action"] == "Force garbage collection":
                # Force garbage collection
                collected = gc.collect()
                logger.info(f"Applied memory optimization: collected {collected} objects")
                
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "type": optimization["type"],
                    "action": optimization["action"],
                    "result": f"Collected {collected} objects",
                    "success": True
                })
                return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization['action']}: {e}")
            self.optimization_history.append({
                "timestamp": time.time(),
                "type": optimization["type"],
                "action": optimization["action"],
                "result": str(e),
                "success": False
            })
        
        return False


# Global instances
_metrics_collector = MetricsCollector()
_health_monitor = HealthMonitor(_metrics_collector)
_performance_optimizer = PerformanceOptimizer(_health_monitor)


async def get_performance_dashboard() -> Dict[str, Any]:
    """Get comprehensive performance dashboard data."""
    metrics = await _metrics_collector.collect_system_metrics()
    health = await _health_monitor.assess_system_health()
    alerts = await _health_monitor.get_alerts()
    suggestions = await _performance_optimizer.suggest_optimizations()
    
    # Get operation statistics
    redis_stats = await _metrics_collector.get_operation_stats("redis")
    dag_stats = await _metrics_collector.get_operation_stats("dag_complete")
    evolution_stats = await _metrics_collector.get_operation_stats("evolution_step")
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": asdict(metrics),
        "health": asdict(health),
        "alerts": alerts,
        "suggestions": suggestions,
        "operation_stats": {
            "redis": redis_stats,
            "dag_processing": dag_stats,
            "evolution": evolution_stats
        },
        "optimization_history": _performance_optimizer.optimization_history[-10:]  # Last 10
    }


# Export commonly used functions
async def record_operation(operation: str, duration: float, success: bool = True):
    """Record an operation for performance tracking."""
    await _metrics_collector.record_operation(operation, duration, success)


async def record_cache_hit(hit: bool):
    """Record cache hit/miss for performance tracking."""
    await _metrics_collector.record_cache_hit(hit)


async def get_system_health() -> SystemHealth:
    """Get current system health status."""
    return await _health_monitor.assess_system_health()


async def auto_optimize():
    """Run automatic optimizations based on current metrics."""
    suggestions = await _performance_optimizer.suggest_optimizations()
    applied = 0
    
    for suggestion in suggestions:
        if suggestion.get("auto_apply", False):
            if await _performance_optimizer.apply_optimization(suggestion):
                applied += 1
    
    return applied 