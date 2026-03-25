from gigaevo.utils.redis import RedisRunConfig


def create_redis_config(config: dict[str, str]) -> RedisRunConfig:
    """Create a RedisRunConfig based on the config."""
    return RedisRunConfig(
        redis_host=config.get("redis_host", "localhost"),
        redis_port=int(config.get("redis_port", 6379)),
        redis_db=int(config.get("redis_db", 0)),
        redis_prefix=config.get("redis_prefix", ""),
        label=config.get("label", ""),
    )
