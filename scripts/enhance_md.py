import os
import asyncio
import logging
from pydantic import BaseModel, Field
import pydantic_ai
from pydantic_ai import Agent, RunContext # Import Agent
from pydantic_ai.exceptions import ModelHTTPError, AgentRunError
from typing import Optional, List, Tuple
from pydantic import ValidationError
import math

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Ensure the necessary library (e.g., 'openai') is installed: pip install openai pydantic-ai
# This script defaults to OpenAI, requiring the OPENAI_API_KEY environment variable.
PRIMARY_PROVIDER = "google-gla" # Change if using a different provider supported by pydantic-ai
FALLBACK_PROVIDER = "openai" # Fallback provider if primary fails
PRIMARY_MODEL = "gemini-2.5-flash-preview-04-17" # Primary model
FALLBACK_MODEL = "gpt-4.1-mini" # Fallback model
# Maximum token size for a chunk (approximate - Gemini and OpenAI have different token counts)
MAX_CHUNK_SIZE = 10000 # Characters per chunk
MAX_RETRIES = 3 # Number of retries per model before switching

# Define Agent Instructions - these guide the overall behavior
AGENT_INSTRUCTIONS = f"""
You are an expert Markdown processor. Your task is to enhance and clean up provided Markdown text according to specific rules.
You will receive Markdown content as user input which may be a full document or a chunk of a larger document. Apply the following transformations rigorously and return *only* the resulting enhanced Markdown text as a plain string.

Transformations:
1.  **Remove Links and URLs:** Eliminate all `https://...` URLs and `[text](url)` Markdown links. Keep only the link text (e.g., `[F23.2](...)` becomes `F23.2`). If the link text itself is just a URL, remove it entirely.
2.  **Clean Heading Hierarchy:** Restructure headings (`#`, `##`, etc.) for logical flow, starting main sections with `##`. Ensure consistency (e.g., `## Category`, `### SubCategory F20.-`, `#### SpecificCode F20.0`). No bolding in headings. Fix inconsistent levels. Add appropriate spacing.
3.  **Remove Plaintext Tags:** Delete HTML tags like `<br>`, `<i>`, etc.
4.  **Replace Simple Tables (<= 2 Columns):** Convert tables with 1 or 2 columns into structured lists (e.g., bullet points or definition lists), preserving content accurately. Keep multi-line cell content associated with its code/title. Leave tables with >2 columns as is, but clean formatting.
    Example Transformation (Input Table -> Output List):
    Input:
    ```markdown
    | Kode | Titel |
    |---|---|
    | .0 | Akute Intoxikation |
    |    | Description line 1.<br>Description line 2. |
    ```
    Output:
    ```markdown
    *   **.0: Akute Intoxikation**
        Description line 1. Description line 2.
    ```
5.  **Preserve Codes (CRITICAL):** Do NOT modify, delete, or alter ANY diagnostic codes (e.g., `F00.-*`, `G30.-†`, `U63.-!`, `F00.0*`, `F01.-`, `F31.81`, `F34.0`, `Fxx.y`, `Fxx.yz`, `.5`, `.6`, `.7`, `[F23.2]`). Preserve ALL special characters (`*`, `†`, `!`, `-`, `.`, `/`, `[]`, `()`) and ranges (`F10-F19`). Codes MUST remain exactly as in the original text.
6.  **Remove Page Numbers:** Delete references like `(S. 209)`, `(S. 202)`, etc. Remove the entire parenthetical expression.
7.  **Consistent Bolding:** Use bolding (`**text**`) sparingly only for standard labels like `Definition:`, `Inkl.:`, `Exkl.:`, `Beachte:`, `Diagnostische Kriterien:`, `Codierhinweis:`, `Bestimme, ob:`. Remove bolding from headings. Ensure minimal, consistent usage.
8.  **Remove Metadata/Navigation:** Delete extraneous text like "Kode-Suche...", "ICD-10-GM Version...", "Kapitel V", "Title:", "URL Source:", "***", `---`, image links. Start output directly with the main content heading. Ensure clean paragraph spacing.

Chunking Awareness:
- If you receive a chunk that appears to be part of a larger document, maintain logical formatting at chunk boundaries
- For chunks, don't worry about incomplete sections at the beginning or end - these will be merged properly with other chunks
- Ensure your output doesn't introduce artifacts or repetitions when merged with other chunks

Output Format (IMPORTANT):
- Return ONLY the enhanced markdown content as plain text
- DO NOT wrap your response in markdown code blocks (```)
- DO NOT include any explanations or notes before or after the enhanced content
- Provide ONLY the processed markdown content with no additional formatting

Pay extreme attention to preserving diagnostic codes exactly. Return *only* the final, cleaned Markdown string.
"""

# --- Helper Functions ---
def split_into_chunks(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """
    Split the markdown text into manageable chunks while preserving structure.
    Try to split at paragraph boundaries when possible.

    Args:
        text: The markdown text to split
        max_chunk_size: Maximum size of each chunk in characters

    Returns:
        A list of text chunks
    """
    # If the text is already small enough, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]

    # Get approximate number of chunks needed
    num_chunks = math.ceil(len(text) / max_chunk_size)
    approximate_chunk_size = len(text) // num_chunks

    # Prefer splitting at paragraph boundaries
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        # Default end is either max chunk size or end of text
        end_idx = min(start_idx + approximate_chunk_size, len(text))

        # If we're not at the end of the text, try to find a better split point
        if end_idx < len(text):
            # Look for double newlines (paragraph breaks) near the approximate end
            paragraph_break = text.find('\n\n', end_idx - 200, end_idx + 200)
            if paragraph_break != -1:
                end_idx = paragraph_break + 2  # Include the newlines
            else:
                # If no paragraph break, look for single newline
                line_break = text.find('\n', end_idx - 100, end_idx + 100)
                if line_break != -1:
                    end_idx = line_break + 1  # Include the newline

        # Add the chunk
        chunks.append(text[start_idx:end_idx])
        start_idx = end_idx

    return chunks

def create_agent(provider: str, model: str) -> Optional[Agent]:
    """
    Creates a Pydantic-AI Agent with the specified provider and model.

    Args:
        provider: The AI provider to use (e.g., 'openai', 'google-gla')
        model: The model name to use

    Returns:
        An initialized Agent instance or None if initialization fails
    """
    try:
        model_identifier = f"{provider}:{model}" if ":" not in model else model
        agent = Agent(
            model=model_identifier,
            # model="openai:gpt-4.1-nano"
            instructions=AGENT_INSTRUCTIONS,
            output_type=str  # Expecting a plain string as the final enhanced markdown
        )
        logging.info(f"Initialized Pydantic-AI Agent with model: {model_identifier}")
        return agent
    except Exception as e:
        logging.error(f"Failed to initialize agent with {provider}:{model}. Error: {e}")
        return None

# --- Initialize Agents ---
primary_agent = None
fallback_agent = None

try:
    # Initialize primary agent
    primary_agent = create_agent(PRIMARY_PROVIDER, PRIMARY_MODEL)

    # Initialize fallback agent
    fallback_agent = create_agent(FALLBACK_PROVIDER, FALLBACK_MODEL)

    if primary_agent is None and fallback_agent is None:
        logging.error("Both primary and fallback agents failed to initialize. Cannot proceed.")

except ImportError as e:
    logging.error(f"Error: Required library not installed. Details: {e}")
    logging.error("Please install the necessary libraries (e.g., 'pip install openai pydantic-ai')")
except Exception as e:
    logging.error(f"Error initializing Pydantic-AI Agents: {e}")
    logging.error("Please ensure API keys/environment variables are set correctly and model names are valid.")

# Function to process a file using the Agent with chunking and fallback
async def enhance_markdown_file(file_path: str, input_markdown: str) -> str:
    """
    Enhances markdown content with chunking support and fallback to different models.

    Args:
        file_path: The path to the markdown file (for logging/context).
        input_markdown: The raw markdown content to process.

    Returns:
        The enhanced markdown string, or an error message string starting with 'Error:'.
    """
    if primary_agent is None and fallback_agent is None:
        return f"Error: No agent instances available for {file_path}."

    basename = os.path.basename(file_path)
    logging.info(f"Starting enhancement process for: {file_path}")

    # Determine if we need to chunk the content
    chunks = split_into_chunks(input_markdown)
    logging.info(f"Split {basename} into {len(chunks)} chunks for processing")

    # Process each chunk with primary and fallback agents
    enhanced_chunks = []

    for i, chunk in enumerate(chunks):
        chunk_result = await process_chunk(file_path, chunk, i, len(chunks))
        if chunk_result.startswith("Error:"):
            return f"Error: Failed to process {file_path} - chunk {i+1}/{len(chunks)} failed: {chunk_result}"
        enhanced_chunks.append(chunk_result)

    # Combine the chunks
    combined_result = "\n\n".join(enhanced_chunks)
    logging.info(f"Successfully combined {len(enhanced_chunks)} chunks for {basename}")

    # Do one final cleaning of the combined result to ensure no markdown tags remain
    final_result = clean_markdown_output(combined_result)

    return final_result

async def process_chunk(file_path: str, chunk_content: str, chunk_index: int, total_chunks: int) -> str:
    """
    Process a single chunk using primary agent with fallback to secondary agent.

    Args:
        file_path: The file path for context and logging
        chunk_content: The content of this particular chunk
        chunk_index: The index of this chunk (0-based)
        total_chunks: Total number of chunks for this file

    Returns:
        Enhanced chunk content or error message
    """
    basename = os.path.basename(file_path)
    chunk_description = f"{basename} (chunk {chunk_index+1}/{total_chunks})"

    # Try primary agent first
    primary_result = await try_process_with_agent(file_path, chunk_content, chunk_index, total_chunks, primary_agent, PRIMARY_MODEL)
    if not primary_result.startswith("Error:"):
        return primary_result

    # If primary agent fails and we have a fallback, try that
    if fallback_agent is not None:
        logging.warning(f"Primary agent failed for {chunk_description}, trying fallback agent")
        fallback_result = await try_process_with_agent(file_path, chunk_content, chunk_index, total_chunks, fallback_agent, FALLBACK_MODEL)
        return fallback_result

    # If no fallback or fallback also failed
    return primary_result  # Return the primary error

def clean_markdown_output(text: str) -> str:
    """
    Clean markdown output by removing code block tags that the model might add.

    Args:
        text: The markdown text from the model

    Returns:
        Cleaned markdown text without code block delimiters
    """
    # Remove markdown code block tags if present
    if text.startswith('```markdown'):
        text = text[len('```markdown'):]
    elif text.startswith('```md'):
        text = text[len('```md'):]
    elif text.startswith('```'):
        text = text[len('```'):]

    # Remove closing code block if present
    if text.endswith('```'):
        text = text[:-3]

    # Trim whitespace at start and end
    text = text.strip()

    return text

async def try_process_with_agent(file_path: str, chunk_content: str, chunk_index: int, total_chunks: int, agent: Agent, model_name: str) -> str:
    """
    Try to process a chunk with the given agent.

    Args:
        file_path: File path for context
        chunk_content: The content to process
        chunk_index: Index of the chunk
        total_chunks: Total chunks in the file
        agent: The agent to use
        model_name: Name of the model (for logging)

    Returns:
        Enhanced content or error message
    """
    basename = os.path.basename(file_path)
    chunk_description = f"{basename} (chunk {chunk_index+1}/{total_chunks})"

    # Construct prompt with chunk information
    chunk_context = "" if total_chunks == 1 else f" This is chunk {chunk_index+1} of {total_chunks}."
    user_prompt = f"""
Please process the following Markdown content extracted from the file '{basename}' according to the rules defined in your instructions.{chunk_context}

```markdown
{chunk_content}
```

Return *only* the resulting enhanced Markdown text without wrapping it in markdown code blocks or any other tags.
"""

    # Try with retries
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Processing {chunk_description} with {model_name} (attempt {attempt+1}/{MAX_RETRIES})")
            result = await agent.run(user_prompt)

            if isinstance(result.output, str):
                logging.info(f"Successfully processed {chunk_description} with {model_name}")
                # Clean the output to remove any markdown code block tags
                cleaned_output = clean_markdown_output(result.output)
                return cleaned_output
            else:
                logging.error(f"Agent returned unexpected output type for {chunk_description}: {type(result.output)}")
                return f"Error: Agent returned unexpected output type for {chunk_description}."

        except (ModelHTTPError, AgentRunError) as e:
            logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES}: Model communication error for {chunk_description}: {e}")
            if attempt == MAX_RETRIES - 1:
                break  # Let the outer handler deal with the final failure
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

        except ValidationError as e:
            logging.error(f"Validation error processing {chunk_description} with {model_name}: {e}")
            # Don't retry on validation errors as they're likely to persist
            return f"Error: Validation error with {model_name} for {chunk_description}. {str(e)}"

        except Exception as e:
            logging.error(f"Error processing {chunk_description} with {model_name}: {type(e).__name__}: {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1:
                break  # Let the outer handler deal with the final failure
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

    return f"Error: Failed to process {chunk_description} with {model_name} after {MAX_RETRIES} attempts."


# Main execution function (async structure)
async def main(file_paths):
    """Processes a list of markdown files using chunking and fallback mechanisms."""
    if primary_agent is None and fallback_agent is None:
        logging.error("Cannot proceed without at least one configured Agent.")
        print("Script cannot run: All agents failed to initialize. Check API keys, model names, and logs.")
        return

    tasks = []
    for file_path in file_paths:
        if not isinstance(file_path, str) or not file_path:
             logging.warning(f"Invalid file path provided: {file_path}")
             continue

        abs_file_path = os.path.abspath(file_path)

        if not os.path.exists(abs_file_path):
            logging.warning(f"File not found, skipping: {abs_file_path}")
            continue
        if not abs_file_path.lower().endswith('.md'):
            logging.warning(f"Skipping non-markdown file: {abs_file_path}")
            continue

        logging.info(f"Queueing file for processing: {abs_file_path}")
        try:
            with open(abs_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Create and add the task using our enhanced processing function
            tasks.append(asyncio.create_task(
                enhance_markdown_file(abs_file_path, markdown_content),
                name=abs_file_path
            ))
        except Exception as e:
            logging.error(f"Error reading file {abs_file_path}: {e}")

    if not tasks:
        logging.info("No valid markdown files found to process.")
        return

    logging.info(f"Processing {len(tasks)} files...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful_files = 0
    failed_files = 0
    for i, result in enumerate(results):
        task_name = tasks[i].get_name()
        base_name = os.path.basename(task_name)

        if isinstance(result, Exception):
            failed_files += 1
            logging.error(f"\n--- Error Processing Task: {base_name} ---")
            logging.error(f"An unexpected error occurred during task execution: {result}", exc_info=result)
            print(f"Failed to process {base_name} due to an internal error.")
        elif isinstance(result, str) and result.startswith("Error:"):
             failed_files += 1
             logging.error(f"\n--- Error Processing: {base_name} ---")
             logging.error(result)
             print(f"Failed to process {base_name}. Check logs for details.")
        elif isinstance(result, str):
            successful_files += 1
            output_dir = os.path.dirname(task_name)
            output_filename = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_enhanced.md")
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(result)
                logging.info(f"Successfully saved enhanced content to '{output_filename}'")
                print(f"Saved enhanced file: {output_filename}")
            except Exception as e:
                 logging.error(f"Error saving enhanced file {output_filename}: {e}")
                 print(f"Error saving enhanced file for {base_name}. Check logs.")
                 failed_files += 1
                 successful_files -= 1
        else:
            failed_files += 1
            logging.error(f"Unknown result type for task {base_name}: {type(result)}")
            print(f"Failed to process {base_name} due to an unknown issue.")

    logging.info(f"Processing complete. Successful: {successful_files}, Failed: {failed_files}")

if __name__ == "__main__":
    # --- Files to Process ---
    files_to_process = [
        "./sources/icd-10/icd-10_f00-f09.md",
        "./sources/icd-10/icd-10_f10-f19.md",
        "./sources/icd-10/icd-10_f20-f29.md",
        "./sources/icd-10/icd-10_f30-f39.md",
        "./sources/icd-10/icd-10_f40-f48.md",
        "./sources/icd-10/icd-10_f50-f59.md",
        "./sources/icd-10/icd-10_f60-f69.md",
        "./sources/icd-10/icd-10_f70-f79.md",
        "./sources/icd-10/icd-10_f80-f89.md",
        "./sources/icd-10/icd-10_f90-f98.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_01.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_02.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_03.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_04.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_05.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_06.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_07.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_08.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_09.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_10.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_11.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_12.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_13.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_14.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_15.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_16.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_17.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_18.md",
        "./sources/dsm-5/2_diagnoses/dsm-5_2_19.md",
    ]

    # Check if at least one agent was initialized before running
    if primary_agent or fallback_agent:
        logging.info("Starting Markdown enhancement process...")
        try:
            asyncio.run(main(files_to_process))
        except RuntimeError as e:
             if "Cannot run the event loop while another loop is running" in str(e):
                 logging.warning("Detected running event loop. Please run this script in an environment where a new asyncio loop can be started.")
                 print("Error: Cannot run asyncio loop. See logs.")
             else:
                 logging.error(f"RuntimeError during execution: {e}", exc_info=True)
                 raise e
        except KeyboardInterrupt:
            logging.info("Process interrupted by user.")
    else:
        print("Script cannot execute due to Agent initialization failures. Check API keys, models, and logs.")
