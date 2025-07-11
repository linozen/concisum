import asyncio
import argparse
import logging
from pathlib import Path

# Note: This module requires asyncpg and openai packages:
# pip install asyncpg openai

from concisum.diagnosis.database import build_diagnosis_database

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Build the diagnosis reference database")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the diagnosis JSON data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-granite-embedding-278m-multilingual",
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default="localhost",
        help="Database host address"
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="Database port"
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="Database username"
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default="postgres",
        help="Database password"
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="concisum_rag",
        help="Database name"
    )

    args = parser.parse_args()

    LOG.info(f"Building diagnosis database with data from {args.data}")

    await build_diagnosis_database(
        diagnosis_data_path=Path(args.data),
        embedding_model=args.model,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name
    )

    LOG.info("Database build complete!")


if __name__ == "__main__":
    asyncio.run(main())
