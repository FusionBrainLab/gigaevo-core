"""GigaEvo monitoring library -- shared Redis queries, snapshots, and alerts.

Import concrete symbols from their submodules (e.g.
``from gigaevo.monitoring.alerts import Alert``). The package root is kept
import-light on purpose: ``import gigaevo`` force-imports
``gigaevo.monitoring.events`` to register canonical events, and eager
re-exports here would drag the Redis/OpenAI/langchain closure into every
leaf tool that only needs the lightweight modules.
"""
