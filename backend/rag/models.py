import pymysql
import os
import datetime
from pathlib import Path

import peewee as pw
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1] if len(BASE_DIR.parents) > 1 else BASE_DIR
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=False)

IS_DOCKER = Path("/.dockerenv").exists()
db_name = os.getenv("MYSQL_DATABASE", "qna")
db_host = os.getenv("MYSQL_HOST") or ("qna-mysql" if IS_DOCKER else "localhost")
db_password = os.getenv("MYSQL_ROOT_PASSWORD")
db_port = int(os.getenv("MYSQL_PORT") or (3306 if IS_DOCKER else 3307))
ACCESS_TOKEN_KEY = os.getenv("ACCESS_TOKEN_KEY", "")

conn = pymysql.connect(host=db_host, port=db_port, user='root', password=db_password)
cursor = conn.cursor()
cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")
result = cursor.fetchall()
if result:
    print("Database exists")
else:
    print("Database not exists")
    cursor.execute(f'CREATE DATABASE {db_name}')
conn.close()

myDB = pw.MySQLDatabase(
    host=db_host,
    port=db_port,
    user="root",
    passwd=db_password,
    database=db_name
)

class MySQLModel(pw.Model):
    """A base model that will use our MySQL database"""
    id = pw.PrimaryKeyField(null=False)
    createdAt = pw.DateTimeField(default=datetime.datetime.now)
    updatedAt = pw.DateTimeField()
    
    def save(self, *args, **kwargs):
        self.updatedAt = datetime.datetime.now()
        return super(MySQLModel, self).save(*args, **kwargs)

    class Meta:
        database = myDB
        legacy_table_names = False

class QuestionModel(MySQLModel):
    email = pw.CharField(50)
    question = pw.TextField()
    response = pw.TextField()
    model = pw.CharField(64, default="", null=True)
    chat_id = pw.IntegerField(null=True, index=True)

class Reference(MySQLModel):
    question_id = pw.ForeignKeyField(QuestionModel)
    mapc = pw.CharField(255)
    noidung = pw.TextField()
    ten = pw.TextField()

myDB.connect()
myDB.create_tables([QuestionModel, Reference])

columns = {column.name for column in myDB.get_columns(QuestionModel._meta.table_name)}
if "model" not in columns:
    myDB.execute_sql(f"ALTER TABLE `{QuestionModel._meta.table_name}` ADD COLUMN `model` VARCHAR(64) NULL")
if "chat_id" not in columns:
    myDB.execute_sql(f"ALTER TABLE `{QuestionModel._meta.table_name}` ADD COLUMN `chat_id` INT NULL")
    myDB.execute_sql(f"UPDATE `{QuestionModel._meta.table_name}` SET `chat_id` = `id` WHERE `chat_id` IS NULL")
elif myDB.execute_sql(f"SELECT COUNT(*) FROM `{QuestionModel._meta.table_name}` WHERE `chat_id` IS NULL").fetchone()[0]:
    myDB.execute_sql(f"UPDATE `{QuestionModel._meta.table_name}` SET `chat_id` = `id` WHERE `chat_id` IS NULL")
