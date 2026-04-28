import redis
from dotenv import load_dotenv
import os


load_dotenv()
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_db = int(os.getenv("REDIS_DB", "0"))
redis_password = os.getenv("REDIS_PASSWORD") or None
redis_ttl_seconds = int(os.getenv("RAG_CACHE_TTL_SECONDS", "86400"))

redisClient = redis.Redis(
    host=redis_host,
    port=redis_port,
    db=redis_db,
    password=redis_password,
    socket_connect_timeout=5,
    socket_timeout=5,
)

if redisClient.ping():
    print('Connected to Redis server')
else:
    print('Failed to connect to Redis server')
