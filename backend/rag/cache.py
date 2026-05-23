import redis
from dotenv import load_dotenv
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1] if len(BASE_DIR.parents) > 1 else BASE_DIR
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
IS_DOCKER = Path("/.dockerenv").exists()
redis_host = os.getenv("REDIS_HOST") or ("redis" if IS_DOCKER else "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_db = int(os.getenv("REDIS_DB", "0"))
redis_password = os.getenv("REDIS_PASSWORD") or None
redis_ttl_seconds = int(os.getenv("REDIS_TTL_SECONDS", "86400"))

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
