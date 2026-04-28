import csv
import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", BASE_DIR.parent / "backend" / "rag" / "corpus")).resolve()
DB_NAME = os.getenv("MYSQL_DATABASE", "law")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456789")
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
BATCH_SIZE = int(os.getenv("MYSQL_IMPORT_BATCH_SIZE", "1000"))
RESET_TABLES = os.getenv("RESET_TABLES", "true").lower() in {"1", "true", "yes"}


field_size_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_size_limit)
        break
    except OverflowError:
        field_size_limit //= 10


def connect(database=None):
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=database,
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_database():
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        connection.commit()


def ensure_schema(connection):
    with connection.cursor() as cursor:
        if RESET_TABLES:
            cursor.execute("DROP TABLE IF EXISTS `vb_chimuc`")
            cursor.execute("DROP TABLE IF EXISTS `pddieu`")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS `pddieu` (
                `id` VARCHAR(128) PRIMARY KEY,
                `title` TEXT NULL,
                `content` LONGTEXT NULL,
                `demuc_id` VARCHAR(128) NULL,
                `chuong_id` VARCHAR(128) NULL,
                INDEX `idx_pddieu_demuc_id` (`demuc_id`),
                INDEX `idx_pddieu_chuong_id` (`chuong_id`)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
    connection.commit()


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def flush_pddieu(connection, rows):
    if not rows:
        return 0
    values = [
        (
            row.get("id"),
            row.get("title"),
            row.get("content"),
            row.get("demuc_id"),
            row.get("chuong_id"),
        )
        for row in rows
        if row.get("id")
    ]
    if not values:
        return 0
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO `pddieu`
                (`id`, `title`, `content`, `demuc_id`, `chuong_id`)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                `title` = VALUES(`title`),
                `content` = VALUES(`content`),
                `demuc_id` = VALUES(`demuc_id`),
                `chuong_id` = VALUES(`chuong_id`)
            """,
            values,
        )
    connection.commit()
    return len(values)


def import_csv(connection, file_name, flush):
    path = CORPUS_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")

    batch = []
    total = 0
    for row in read_csv_rows(path):
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            total += flush(connection, batch)
            print(f"Imported {total} rows from {file_name}")
            batch.clear()

    total += flush(connection, batch)
    print(f"Done importing {total} rows from {file_name}")
    return total


def main():
    ensure_database()
    with connect(DB_NAME) as connection:
        ensure_schema(connection)
        pddieu_total = import_csv(connection, "pddieu.csv", flush_pddieu)
        print(f"Imported corpus into MySQL database `{DB_NAME}`")
        print(f"- pddieu: {pddieu_total} rows")


if __name__ == "__main__":
    main()
