import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import os


# Setup logging
LOG = logging.getLogger(__name__)


async def setup_database(
    db_host="localhost",
    db_port=5432,
    db_user="postgres",
    db_password="postgres",
    db_name="concisum_rag",
):
    """Set up the database with pgvector extension."""
    import asyncpg

    # Database schema definition
    db_schema = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS diagnosis_references (
        id serial PRIMARY KEY,
        code text NOT NULL,
        title text NOT NULL,
        category text NOT NULL,
        description_short text NOT NULL,
        symptoms_keywords jsonb NOT NULL,
        criteria_text text NOT NULL,
        diagnostic_features_text text NOT NULL,
        severity_levels jsonb,
        differential_diagnosis_codes jsonb,
        source_document text,
        embedding_general vector(768) NOT NULL,
        embedding_symptoms vector(768)
    );
    CREATE INDEX IF NOT EXISTS idx_diagnosis_embedding_general ON diagnosis_references USING hnsw (embedding_general vector_l2_ops);
    CREATE INDEX IF NOT EXISTS idx_diagnosis_embedding_symptoms ON diagnosis_references USING hnsw (embedding_symptoms vector_l2_ops);
    """

    # Connection string
    server_dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}"

    # Create database if it doesn't exist
    conn = await asyncpg.connect(server_dsn)
    try:
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not db_exists:
            LOG.info(f"Creating database {db_name}")
            await conn.execute(f"CREATE DATABASE {db_name}")
    finally:
        await conn.close()

    # Connect to the database and create schema
    pool = await asyncpg.create_pool(f"{server_dsn}/{db_name}")
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(db_schema)
        LOG.info("Database schema created or already exists")
    finally:
        await pool.close()


async def generate_embedding(
    text: str, model_name: str = "text-embedding-granite-embedding-278m-multilingual"
) -> List[float]:
    """Generate embeddings for a given text."""
    # Import here to avoid failures if not installed
    from openai import OpenAI

    # Use LM Studio
    api_key = "lm_studio"
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)
    embedding_resp = client.embeddings.create(
        input=text,
        model=model_name,
    )
    return embedding_resp.data[0].embedding


async def build_diagnosis_database(
    diagnosis_data_path: Path,
    embedding_model: str = "text-embedding-3-small",
    db_host: str = "localhost",
    db_port: int = 5432,
    db_user: str = "postgres",
    db_password: str = "postgres",
    db_name: str = "concisum_rag",
):
    """Build the diagnosis reference database."""
    import asyncpg
    import pydantic_core

    # Setup database
    await setup_database(db_host, db_port, db_user, db_password, db_name)

    # Connection string
    server_dsn = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Process diagnosis data
    if diagnosis_data_path.exists():
        with open(diagnosis_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            diagnosis_data = data.get("diagnoses", [])

        # Connect to database
        pool = await asyncpg.create_pool(server_dsn)
        try:
            for entry in diagnosis_data:
                code = entry.get("code")
                # Check if entry already exists
                exists = await pool.fetchval(
                    "SELECT 1 FROM diagnosis_references WHERE code = $1", code
                )

                if exists:
                    LOG.info(f"Skipping existing diagnosis entry: {code}")
                    continue

                # Check for required fields
                required_fields = [
                    "code",
                    "title",
                    "category",
                    "description_short",
                    "criteria_text",
                ]
                missing_fields = [
                    field for field in required_fields if not entry.get(field)
                ]
                if missing_fields:
                    LOG.error(
                        f"Missing required fields for entry {code}: {', '.join(missing_fields)}"
                    )
                    continue

                # Prepare general embedding content for overall diagnosis information
                general_embedding_content = f"""Code: {entry.get('code')}
                Title: {entry.get('title')}
                Category: {entry.get('category')}
                Description: {entry.get('description_short')}
                Diagnostic Features: {entry.get('diagnostic_features_text', '')}
                Criteria: {entry.get('criteria_text')}"""

                # Prepare symptoms-specific embedding content
                symptoms_keywords = entry.get("symptoms_keywords", [])
                symptoms_embedding_content = (
                    "\n".join(symptoms_keywords) if symptoms_keywords else None
                )

                # Generate embeddings
                try:
                    # General diagnosis embedding
                    general_embedding = await generate_embedding(
                        general_embedding_content, embedding_model
                    )
                    general_embedding_json = pydantic_core.to_json(
                        general_embedding
                    ).decode()

                    # Symptoms embedding (if available)
                    symptoms_embedding_json = None
                    if symptoms_embedding_content:
                        symptoms_embedding = await generate_embedding(
                            symptoms_embedding_content, embedding_model
                        )
                        symptoms_embedding_json = pydantic_core.to_json(
                            symptoms_embedding
                        ).decode()

                    # Prepare JSON fields
                    symptoms_keywords_json = pydantic_core.to_json(
                        entry.get("symptoms_keywords", [])
                    ).decode()
                    severity_levels_json = pydantic_core.to_json(
                        entry.get("severity_levels", [])
                    ).decode()
                    differential_diagnosis_codes_json = pydantic_core.to_json(
                        entry.get("differential_diagnosis_codes", [])
                    ).decode()

                    # Insert into database
                    await pool.execute(
                        """INSERT INTO diagnosis_references
                        (code, title, category, description_short, symptoms_keywords,
                        criteria_text, diagnostic_features_text, severity_levels,
                        differential_diagnosis_codes, source_document,
                        embedding_general, embedding_symptoms)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                        entry.get("code"),
                        entry.get("title"),
                        entry.get("category"),
                        entry.get("description_short"),
                        symptoms_keywords_json,
                        entry.get("criteria_text"),
                        entry.get("diagnostic_features_text", ""),  # Optional field
                        severity_levels_json,
                        differential_diagnosis_codes_json,
                        entry.get("source_document", None),  # Optional field
                        general_embedding_json,
                        symptoms_embedding_json,
                    )
                    LOG.info(f"Added diagnosis entry: {code}")
                except Exception as e:
                    LOG.error(f"Error processing diagnosis entry {code}: {str(e)}")
        finally:
            await pool.close()


async def retrieve_diagnoses(
    query: str,
    symptoms_mode: bool = False,
    limit: int = 3,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_user: str = "postgres",
    db_password: str = "postgres",
    db_name: str = "concisum_rag",
) -> List[Dict[str, Any]]:
    """Retrieve relevant diagnoses based on a search query.

    Args:
        query: The search query
        symptoms_mode: If True, search against symptoms embeddings instead of general embeddings
        limit: Maximum number of results to return
    """
    try:
        import asyncpg
        import pydantic_core

        # Generate embedding for the query
        embedding = await generate_embedding(query)
        embedding_json = pydantic_core.to_json(embedding).decode()

        # Connection string
        server_dsn = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        # Connect to database and search for similar diagnoses
        pool = await asyncpg.create_pool(server_dsn)
        try:
            if symptoms_mode:
                # Search based on symptoms similarity
                rows = await pool.fetch(
                    """SELECT
                        code, title, category, description_short,
                        symptoms_keywords, criteria_text, diagnostic_features_text,
                        severity_levels, differential_diagnosis_codes, source_document
                    FROM diagnosis_references
                    WHERE embedding_symptoms IS NOT NULL
                    ORDER BY embedding_symptoms <-> $1
                    LIMIT $2""",
                    embedding_json,
                    limit,
                )
            else:
                # Search based on general information similarity
                rows = await pool.fetch(
                    """SELECT
                        code, title, category, description_short,
                        symptoms_keywords, criteria_text, diagnostic_features_text,
                        severity_levels, differential_diagnosis_codes, source_document
                    FROM diagnosis_references
                    ORDER BY embedding_general <-> $1
                    LIMIT $2""",
                    embedding_json,
                    limit,
                )

            # Convert JSONB fields back to Python objects
            results = []
            for row in rows:
                row_dict = dict(row)
                # Parse JSON fields back to Python objects
                for field in [
                    "symptoms_keywords",
                    "severity_levels",
                    "differential_diagnosis_codes",
                ]:
                    if row_dict.get(field):
                        row_dict[field] = json.loads(row_dict[field])
                    else:
                        row_dict[field] = None
                results.append(row_dict)

            return results
        finally:
            await pool.close()
    except Exception as e:
        LOG.error(f"Error retrieving diagnoses: {str(e)}")
        return []
