from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from os import getenv
TOPIC_DB_PATH= getenv("TOPIC_DB_PATH", "chroma_db_law") #Chroma DB Persist Directory for TOPIC
ST_MODEL_PATH= getenv("ST_MODEL_PATH", "keepitreal/vietnamese-sbert") #Sentence Transformers Model Path
ENVIRONMENT = getenv("ENVIRONMENT")
ACCESS_TOKEN_KEY= getenv("ACCESS_TOKEN_KEY")
if ENVIRONMENT == "production":
    with open("/run/secrets/access_token_key", "r") as access_token_file:
        ACCESS_TOKEN_KEY = access_token_file.read()
    access_token_file.close()


