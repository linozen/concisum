import json
import re
import asyncio
import logging
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

"""
This script uses pydantic-ai to parse detailed Markdown descriptions of medical diagnoses
(like those found in DSM-5 or ICD formats) and extract structured information into a JSON file.
This JSON format is optimized for populating a vector database for Retrieval-Augmented Generation (RAG)
based on symptom queries.

The script identifies individual diagnosis sections within Markdown files,
uses an LLM Agent to extract key information according to a Pydantic model,
and aggregates the results into a single JSON output file.

Each diagnosis entry in the output JSON will adhere to the 'DiagnosisOutput' Pydantic model.
"""

# --- Configuration ---
OVERVIEW_DIR = Path("data/refined/0_overview")  # Directory containing overview files with codes and titles
DIAGNOSES_DIR = Path("data/refined/2_diagnoses")  # Directory containing detailed diagnosis chapters
OUTPUT_FILE = Path("data/diagnoses_extracted.json")
LOG_FILE = Path("logs/extraction.log")
LOG_LEVEL = logging.INFO

# Temporary storage for pre-filled diagnoses from overview files
DIAGNOSES_CACHE_FILE = Path("data/diagnosis_cache.json")

# Choose a capable model for extraction. gpt-4o / claude-3.5-sonnet recommended.
LLM_MODEL = "google-gla:gemini-2.0-flash"

# Create logfile directory if it doesn't exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("diagnosis_extractor")

# Regex patterns for diagnosis identification and content parsing
# More robust pattern to handle various heading formats with diagnosis titles and codes
# It captures the heading level (###), title (which may or may not be in bold **Title** format),
# and an optional code (Fxx.x, Fxx, etc.)
DIAGNOSIS_SPLIT_REGEX = r"^(#+)\s+(?:\*\*)?([^\*]+?)(?:\*\*)?\s*(?:([A-Z][0-9]{2}(?:\.[0-9]+)?(?:-[0-9]+)?))?\s*$"

# Alternative patterns to try if the main pattern doesn't match (more permissive)
ALT_DIAGNOSIS_PATTERNS = [
    r"^(#+)\s+([^\(\)]+)\s*\(?\s*([A-Z][0-9]{2}(?:\.[0-9]+)?)\s*\)?\s*$",  # Title (Code) format
    r"^(#+)\s+(?:\*\*)?([^:]+?)(?:\*\*)?:\s*([A-Z][0-9]{2}(?:\.[0-9]+)?)\s*$",  # Title: Code format
    r"^(#+)\s+([^\(\)]+)\s*\(([^\)]+)\)\s*$"  # Title (anything in parentheses) format
]

# Pattern to identify category headings (typically higher level headings)
CATEGORY_HEADING_REGEX = r"^(#+)\s+(?:\*\*)?([^\*]+?)(?:\*\*)?\s*$"

# --- Pydantic Model for Structured Output ---

class DiagnosisOutput(BaseModel):
    """Pydantic model defining the structured output for each diagnosis."""
    code: Optional[str] = Field(None, description="The ICD-10 or DSM code (e.g., 'F84.0', 'F80.2'). Extracted from the heading if possible, otherwise from the text.")
    title: str = Field(..., description="The official title or name of the diagnosis (e.g., 'Autismus-Spektrum-Störung').")
    category: Optional[str] = Field(None, description="The higher-level category the diagnosis belongs to (e.g., 'Störungen der neuronalen und mentalen Entwicklung'). Extracted from the nearest preceding H1/H2 heading.")
    description_short: str = Field(..., description="A brief, concise 1-2 sentence summary of the disorder based on the initial paragraphs of its section.")
    symptoms_keywords: List[str] = Field(..., description="A list of specific, concise German keywords/phrases representing defining symptoms, deficits, or characteristic behaviors mentioned in criteria and features sections.")
    criteria_text: Optional[str] = Field(None, description="The full text detailing the diagnostic criteria (e.g., sections starting with 'Diagnostische Kriterien' or 'A.', 'B.', 'C.').")
    diagnostic_features_text: Optional[str] = Field(None, description="Detailed text describing the characteristic features and associated aspects ('Diagnostische Merkmale', 'Zugehörige Merkmale').")
    severity_levels: Optional[str] = Field(None, description="Text describing different severity specifiers (e.g., mild, moderate, severe), including criteria or typical presentations for each level. Include tables formatted as text if present.")
    differential_diagnosis_text: Optional[str] = Field(None, description="Text describing differential diagnoses.")
    comorbidity_text: Optional[str] = Field(None, description="Text describing comorbidities.")
    source_document: str = Field(..., description="The path to the source markdown file.")

# --- Pydantic AI Agent Definition ---

# Configure the agent responsible for extracting information
extraction_agent = Agent(
    LLM_MODEL,
    output_type=DiagnosisOutput,
    instructions="""
        You are an expert medical information extractor specializing in psychiatric diagnostic manuals like DSM-5.
        You will be given a Markdown text block containing information about ONE specific medical diagnosis.
        Your task is to carefully read the text and extract the required information to populate ALL fields of the 'DiagnosisOutput' JSON schema accurately and comprehensively.
        Pay close attention to extracting specific diagnostic criteria, defining features, and a comprehensive list of symptom keywords.
        For 'symptoms_keywords', list specific, distinct German phrases or keywords representing observable behaviors, deficits, core symptoms, or characteristic experiences mentioned in the criteria and features/description sections. Be granular and capture variations (e.g., 'langsames Lesen', 'mühsames Lesen').
        For 'criteria_text', extract the complete section detailing the diagnostic criteria (often starting with 'Diagnostische Kriterien' or marked A, B, C...).
        For 'diagnostic_features_text', extract the narrative descriptions under headings like 'Diagnostische Merkmale' or 'Zugehörige Merkmale'.
        For 'severity_levels', extract any text describing severity, including tables. Format tables clearly within the text.
        For 'differential_diagnosis_text', extract the content of the 'Differenzialdiagnose' section.
        For 'comorbidity_text', extract the content of the 'Komorbidität' section.
        For 'category', identify the most relevant higher-level category heading (usually H1 or H2, like '# **Störungen der neuronalen und mentalen Entwicklung**') that precedes this diagnosis section.
        Determine the 'code' (like Fxx.x) and 'title' primarily from the main heading of the provided text block.
        Generate a concise 'description_short' (1-2 sentences) summarizing the disorder from the introductory text for that specific diagnosis.
        Populate 'source_document' with the provided filename.
        If a specific section (like 'severity_levels' or 'differential_diagnosis_text') is not present in the text for this specific diagnosis, set the corresponding field to null or omit it if optional, but try to populate all fields where information exists.
        Output ONLY the single, valid JSON object conforming to the 'DiagnosisOutput' schema. Do not include any explanations or markdown formatting around the JSON.
    """
)

# --- Helper Classes and Functions ---

@dataclass
class DiagnosisEntry:
    """Basic entry for a diagnosis with essential information."""
    code: str
    title: str
    category: Optional[str] = None
    description_short: Optional[str] = None
    symptoms_keywords: List[str] = field(default_factory=list)
    criteria_text: Optional[str] = None
    diagnostic_features_text: Optional[str] = None
    severity_levels: Optional[str] = None
    differential_diagnosis_text: Optional[str] = None
    comorbidity_text: Optional[str] = None
    source_document: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the entry to a dictionary for JSON serialization."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if getattr(self, field) is not None
        }

    def update_from_dict(self, data: dict) -> None:
        """Update the entry from a dictionary, skipping None values."""
        for key, value in data.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

@dataclass
class MarkdownSection:
    """Represents a section of a markdown file with diagnosis information."""
    title: str
    code: Optional[str] = None
    category: Optional[str] = None
    heading_level: int = 0
    content: str = ""
    heading_line: str = ""
    line_number: int = 0
    source_file: str = ""

    def is_valid_diagnosis(self) -> bool:
        """Check if this section appears to be a valid diagnosis section."""
        # Basic validation - must have a title
        if not self.title or not self.content:
            return False
            
        # Relaxed content length validation - 50 chars instead of 100
        if len(self.content) < 50:
            return False

        # Look for key diagnostic indications in the content
        diagnostic_indicators = [
            "diagnos", "störung", "syndrom", "kriterien", "symptom",
            "merkmale", "schweregrad", "komorbid", "differenzialdiagnos",
            "episode", "zugehörige", "risiko", "prävalenz", "verlauf"
        ]

        # If it has an ICD/DSM-style code, it's very likely a diagnosis
        if self.code and re.match(r"[A-Z][0-9]{2}(?:\.[0-9]+)?", self.code):
            return True

        # If the section title contains diagnostic terms, it's likely a diagnosis
        diagnostic_title_terms = ["störung", "syndrom", "episode", "diagnos", "depression"]
        if any(term in self.title.lower() for term in diagnostic_title_terms):
            return True

        # Check for diagnostic indicators in the content
        return any(indicator in self.content.lower() for indicator in diagnostic_indicators)

    def get_metadata_dict(self) -> dict:
        """Return a dictionary of metadata about this section."""
        return {
            "title": self.title,
            "code": self.code,
            "category": self.category,
            "heading_level": self.heading_level,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "content_length": len(self.content),
            "is_valid": self.is_valid_diagnosis()
        }

def try_extract_heading_components(line: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Try to extract heading level, title, and code using multiple patterns."""
    # First try the primary regex pattern
    match = re.match(DIAGNOSIS_SPLIT_REGEX, line)
    if match:
        heading_level, title, code = match.groups()
        return len(heading_level), title.strip(), code.strip() if code else None

    # Try alternative patterns
    for pattern in ALT_DIAGNOSIS_PATTERNS:
        match = re.match(pattern, line)
        if match:
            heading_level, title, code = match.groups()
            return len(heading_level), title.strip(), code.strip() if code else None

    # Try to extract just heading level and title (no code)
    category_match = re.match(CATEGORY_HEADING_REGEX, line)
    if category_match:
        heading_level, title = category_match.groups()
        return len(heading_level), title.strip(), None

    # If nothing matches, this is not a heading
    return None, None, None

def is_diagnosis_heading(line: str, min_heading_level: int = 3, max_heading_level: int = 6) -> bool:
    """Determine if a line is likely a diagnosis heading based on patterns and context."""
    level, title, code = try_extract_heading_components(line)

    if level is None or not title:
        return False

    # If it has an appropriate heading level and either a code or diagnostic-looking title
    if min_heading_level <= level <= max_heading_level:
        # If there's a code, assume it's a diagnosis
        if code and re.match(r"[A-Z][0-9]{2}(?:\.[0-9]+)?", code):
            return True

        # Check if title has diagnostic keywords
        diagnostic_terms = ["störung", "syndrom", "erkrankung", "diagnose", "krankheit"]
        return any(term in title.lower() for term in diagnostic_terms)

    return False

def is_category_heading(line: str, max_category_level: int = 2) -> bool:
    """Determine if a line is likely a category heading (higher-level sections)."""
    level, title, _ = try_extract_heading_components(line)

    # Categories are typically higher level headings (H1, H2)
    return level is not None and level <= max_category_level and title is not None

def find_nearest_category(lines: List[str], current_index: int) -> Tuple[Optional[str], Optional[int]]:
    """Find the nearest category heading above the current position."""
    for i in range(current_index - 1, -1, -1):
        line = lines[i].strip()
        if is_category_heading(line):
            level, title, _ = try_extract_heading_components(line)
            # Clean up markdown formatting and asterisks
            if title:
                title = re.sub(r'\*\*|\*', '', title).strip()
            return title, level

    return None, None

def split_markdown_by_diagnosis(content: str, source_file: str = "") -> List[MarkdownSection]:
    """Splits markdown content into sections based on diagnosis headings."""
    sections = []
    current_section = None
    category_cache = {}  # Cache categories by level
    lines = content.splitlines()
    current_category = None
    current_category_level = None

    # Initial scan to identify categories
    for i, line in enumerate(lines):
        if is_category_heading(line):
            level, title, _ = try_extract_heading_components(line)
            if title:
                title = re.sub(r'\*\*|\*', '', title).strip()
                category_cache[level] = title
                if not current_category or (level is not None and current_category_level is not None and level < current_category_level):
                    current_category = title
                    current_category_level = level
                logger.debug(f"Found category: {title} (level {level})")

    # Main parsing pass
    for i, line in enumerate(lines):
        # Try to detect if this is a diagnosis heading
        level, title, code = try_extract_heading_components(line)

        # If this appears to be a diagnosis heading - more permissive conditions
        if level and title and (
            is_diagnosis_heading(line) or 
            (level >= 2 and code and re.match(r"[A-Z][0-9]{2}(?:\.[0-9]+)?", code or "")) or
            (level >= 2 and any(term in title.lower() for term in ["störung", "depression", "diagnose", "episode", "syndrom"]))  
        ):
            # Save previous section if it exists
            if current_section:
                sections.append(current_section)

            # Find the most appropriate category for this diagnosis
            section_category = current_category

            # Try to find a more specific category (usually one level up from diagnosis)
            if level and level > 1:
                for parent_level in range(level - 1, 0, -1):
                    if parent_level in category_cache:
                        section_category = category_cache[parent_level]
                        break

            # Clean the title (remove markdown formatting)
            title = re.sub(r'\*\*|\*', '', title).strip()

            # Create new section
            current_section = MarkdownSection(
                title=title,
                code=code,
                category=section_category,
                heading_level=level,
                content=line,  # Initial content is just the heading line
                heading_line=line,
                line_number=i + 1,
                source_file=source_file
            )

            logger.debug(f"Found diagnosis section: {title} ({code}) - level {level}, category: {section_category}")

        elif is_category_heading(line):
            # Update current category if we encounter a category heading
            level, title, _ = try_extract_heading_components(line)
            if title:
                title = re.sub(r'\*\*|\*', '', title).strip()
                category_cache[level] = title

                # Only update the current category if this is a higher level than current
                if current_category_level is None or (level and level <= current_category_level):
                    current_category = title
                    current_category_level = level

        elif current_section:
            # Append to the current section's content
            current_section.content += "\n" + line

    # Add the last section
    if current_section:
        sections.append(current_section)

    # Filter out sections that are likely not valid diagnoses
    valid_sections = [section for section in sections if section.is_valid_diagnosis()]

    # Log statistics
    logger.info(f"Found {len(valid_sections)} valid diagnosis sections out of {len(sections)} total sections")

    return valid_sections

# --- Main Execution Logic ---

def process_file(md_file_path: Path) -> List[dict]:
    """Process a single markdown file and extract diagnoses."""
    file_diagnoses = []
    logger.info(f"Processing file: {md_file_path.name}...")

    try:
        # Read the file content with UTF-8 encoding
        try:
            content = md_file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            logger.warning(f"UTF-8 decoding failed for {md_file_path.name}, trying latin-1...")
            content = md_file_path.read_text(encoding="latin-1")
            
        # Extract the chapter title to use as category information
        chapter_title = extract_chapter_title(content)
        if chapter_title:
            logger.info(f"  Chapter title: {chapter_title}")
        
        # Split the file into diagnosis sections
        diagnosis_sections = split_markdown_by_diagnosis(content, md_file_path.name)
        
        # If no sections were found, try a more aggressive approach
        if len(diagnosis_sections) == 0:
            logger.warning(f"  No sections found with standard approach, trying alternative method...")
            # Try to extract major headers as sections
            lines = content.splitlines()
            current_section = None
            level_threshold = 3  # Consider headers up to level 3
            
            for i, line in enumerate(lines):
                level, title, code = try_extract_heading_components(line)
                if level and level <= level_threshold and title:
                    # This looks like a major section header
                    if current_section:
                        diagnosis_sections.append(current_section)
                    
                    current_section = MarkdownSection(
                        title=title,
                        code=code,
                        category=chapter_title,
                        heading_level=level,
                        content="",
                        heading_line=line,
                        line_number=i + 1,
                        source_file=md_file_path.name
                    )
                elif current_section:
                    current_section.content += line + "\n"
            
            # Add the last section
            if current_section:
                diagnosis_sections.append(current_section)
                
            # Log the results of the alternative approach
            logger.info(f"  Alternative approach found {len(diagnosis_sections)} potential sections")
            
        logger.info(f"  Found {len(diagnosis_sections)} valid diagnosis sections in {md_file_path.name}")

        # Process each diagnosis section
        for i, section in enumerate(diagnosis_sections):
            logger.info(f"    Processing section {i+1}/{len(diagnosis_sections)}: '{section.title}' ({section.code or 'No code'})...")

            # Skip sections that are too small or likely not diagnoses
            if not section.is_valid_diagnosis():
                logger.warning(f"      Skipping section '{section.title}' - doesn't appear to be a valid diagnosis section")
                continue

            # Create a prompt with additional context
            prompt_content = (
                f"Source Filename: {md_file_path.name}\n\n"
                f"Heading: {section.heading_line}\n\n"
                f"Category: {section.category or 'Unknown'}\n\n"
                f"---\n\n"
                f"{section.content}"
            )

            # Extract structured data using the LLM agent
            try:
                result = extraction_agent.run_sync(prompt_content)
                extracted_data: DiagnosisOutput = result.output

                # Enhance with data from our parsing
                extracted_data.source_document = str(md_file_path.name)

                # Use parsed metadata if LLM missed it
                if not extracted_data.category and section.category:
                    extracted_data.category = section.category
                if not extracted_data.code and section.code:
                    extracted_data.code = section.code

                # Add to our results
                file_diagnoses.append(extracted_data.model_dump())
                logger.info(f"      Successfully extracted: {extracted_data.code or 'No code'} - {extracted_data.title}")

            except UnexpectedModelBehavior as e:
                logger.error(f"      ERROR: Agent failed for section '{section.title}' in {md_file_path.name}: {str(e)}")
            except Exception as e:
                logger.error(f"      ERROR: Unexpected error processing section '{section.title}' in {md_file_path.name}: {str(e)}")

    except Exception as e:
        logger.error(f"  ERROR: Failed to process file {md_file_path.name}: {str(e)}")

    return file_diagnoses

def validate_and_save_output(all_diagnoses: List[dict]) -> None:
    """Validate and save the extracted diagnoses to JSON file."""
    if not all_diagnoses:
        logger.warning("No diagnoses were successfully extracted.")
        return

    logger.info(f"Writing {len(all_diagnoses)} extracted diagnoses to {OUTPUT_FILE}...")

    try:
        # Create output directory if it doesn't exist
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata to the output
        output_data = {
            "metadata": {
                "timestamp": asyncio.run(get_timestamp()),
                "source_files": list(set(item["source_document"] for item in all_diagnoses if "source_document" in item)),
                "diagnosis_count": len(all_diagnoses),
                "extraction_model": LLM_MODEL
            },
            "diagnoses": all_diagnoses
        }

        # Write JSON with proper UTF-8 encoding
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info("Successfully wrote JSON output.")

    except Exception as e:
        logger.error(f"ERROR: Failed to write output JSON file: {str(e)}")

async def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.datetime.now().isoformat()

def extract_diagnoses_from_overview() -> Dict[str, DiagnosisEntry]:
    """Extract diagnosis codes and titles from overview files.

    Returns a dictionary mapping ICD codes to DiagnosisEntry objects.
    Uses multiple strategies to extract from DSM-5 formatted overview files.
    """
    diagnoses_by_code = {}
    overview_files = list(OVERVIEW_DIR.glob("*.md"))

    if not overview_files:
        logger.warning(f"No overview files found in {OVERVIEW_DIR}")
        return diagnoses_by_code

    logger.info(f"Found {len(overview_files)} overview file(s) to process")

    # Process each overview file
    for file_path in overview_files:
        logger.info(f"Processing overview file: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8")
            current_category = None

            # Extract categories and diagnoses
            extract_dsm5_format(content, file_path.name, diagnoses_by_code)

            # Additional fallback pattern to catch any missed diagnoses
            # This is a more general pattern for codes and titles
            fallback_pattern = re.compile(r'\b([A-Z][0-9]{2}(?:\.[0-9]+)?(?:-[0-9]+)?)\b[\s\-]*([^\n\(\)]+)')
            fallback_matches = fallback_pattern.findall(content)

            for code, title in fallback_matches:
                code = code.strip()
                title = title.strip()

                # Skip entries with empty titles or codes
                if not code or not title or len(title) < 3:
                    continue

                # Only add if not already in our dictionary
                if code not in diagnoses_by_code:
                    diagnoses_by_code[code] = DiagnosisEntry(
                        code=code,
                        title=title,
                        source_document=file_path.name,
                        category=current_category
                    )
                    logger.debug(f"Found additional diagnosis via fallback: {code} - {title}")

            logger.info(f"Total diagnoses extracted from {file_path.name}: {sum(1 for d in diagnoses_by_code.values() if d.source_document == file_path.name)}")

        except Exception as e:
            logger.error(f"Error processing overview file {file_path.name}: {str(e)}")

    logger.info(f"Total unique diagnoses extracted from overview files: {len(diagnoses_by_code)}")
    return diagnoses_by_code

def extract_dsm5_format(content: str, source_file: str, diagnoses_dict: Dict[str, DiagnosisEntry]) -> None:
    """Extracts diagnoses from DSM-5 formatted content with specific focus on list formats.

    The DSM-5 format typically includes headers for categories and different formats for diagnoses:
    - Bullet points with codes and titles
    - Indented lists with codes and titles
    - Tables with codes and titles
    """
    # Track the current category as we parse through headings
    current_category = None
    lines = content.splitlines()

    # Pattern to identify category headings (usually with ** or # markers)
    category_pattern = re.compile(r'^#+\s+(?:\*\*)?([^\*]+?)(?:\*\*)?\s*$')

    # Various patterns to capture diagnosis entries in different formats
    # Pattern for bullet points: "- **F32.0** Leichte depressive Episode"
    bullet_pattern = re.compile(r'[-\*]\s+(?:\*\*)?([A-Z][0-9]{2}(?:\.[0-9]+)?(?:-[0-9]+)?)(?:\*\*)?\s+(.+)')

    # Pattern for indented or table entries: "F32.0  Leichte depressive Episode"
    indented_pattern = re.compile(r'^\s*(?:\|\s*)?(?:\*\*)?([A-Z][0-9]{2}(?:\.[0-9]+)?(?:-[0-9]+)?)(?:\*\*)?\s+(.+?)(?:\s*\|)?$')

    # Pattern for entries with brackets: "(F32.0) Leichte depressive Episode"
    bracket_pattern = re.compile(r'\(([A-Z][0-9]{2}(?:\.[0-9]+)?(?:-[0-9]+)?)\)\s*(.+)')

    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue

        # Check if this is a category heading
        category_match = category_pattern.match(line)
        if category_match:
            current_category = category_match.group(1).strip()
            logger.debug(f"Found category: {current_category}")
            continue

        # Try all patterns to extract diagnosis codes and titles
        for pattern in [bullet_pattern, indented_pattern, bracket_pattern]:
            match = pattern.search(line)
            if match:
                code, title = match.groups()
                code = code.strip()
                title = title.strip()

                # Clean up the title (remove any remaining markdown markers)
                title = re.sub(r'\*\*|\*|_', '', title)

                # Skip entries with empty titles or codes or very short titles
                if not code or not title or len(title) < 3:
                    continue

                # Process "Bestimme, ob:" and similar annotations differently
                if title.lower().startswith("bestimme") or title.lower().startswith("beachte"):
                    continue

                # Create or update the diagnosis entry
                if code not in diagnoses_dict:
                    diagnoses_dict[code] = DiagnosisEntry(
                        code=code,
                        title=title,
                        category=current_category,
                        source_document=source_file
                    )
                    logger.debug(f"Found diagnosis: {code} - {title}")
                # If we already have this code but no category, update it
                elif diagnoses_dict[code].category is None and current_category is not None:
                    diagnoses_dict[code].category = current_category
                    logger.debug(f"Updated diagnosis category: {code} - {current_category}")
                # If we have a longer, more descriptive title, use it
                elif len(title) > len(diagnoses_dict[code].title) * 1.5:
                    diagnoses_dict[code].title = title
                    logger.debug(f"Updated diagnosis with longer title: {code} - {title}")

                break  # Found a match, no need to check other patterns

def save_diagnoses_cache(diagnoses: Dict[str, DiagnosisEntry]) -> None:
    """Save the diagnoses cache to a JSON file."""
    try:
        # Convert diagnoses to dictionaries
        diagnoses_dict = {code: entry.to_dict() for code, entry in diagnoses.items()}

        # Create parent directory if it doesn't exist
        DIAGNOSES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(DIAGNOSES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(diagnoses_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(diagnoses)} diagnoses to cache file")
    except Exception as e:
        logger.error(f"Error saving diagnoses cache: {str(e)}")

def load_diagnoses_cache() -> Dict[str, DiagnosisEntry]:
    """Load the diagnoses cache from the JSON file."""
    diagnoses = {}

    if not DIAGNOSES_CACHE_FILE.exists():
        logger.warning(f"Diagnoses cache file not found: {DIAGNOSES_CACHE_FILE}")
        return diagnoses

    try:
        with open(DIAGNOSES_CACHE_FILE, 'r', encoding='utf-8') as f:
            diagnoses_dict = json.load(f)

        # Convert dictionaries back to DiagnosisEntry objects
        for code, entry_dict in diagnoses_dict.items():
            diagnoses[code] = DiagnosisEntry(**entry_dict)

        logger.info(f"Loaded {len(diagnoses)} diagnoses from cache file")
    except Exception as e:
        logger.error(f"Error loading diagnoses cache: {str(e)}")

    return diagnoses



def process_chapter_file(file_path: Path, diagnoses: Dict[str, DiagnosisEntry]) -> None:
    """Process a chapter file and extract detailed information for diagnoses.
    
    This function handles DSM-5 format chapter files, which have inconsistent header levels
    and need special handling to correctly identify diagnosis sections.
    """
    # Log more detailed information for diagnosis extraction
    logger.debug(f"Starting detailed processing of {file_path.name}")
    try:
        # Read the file content with UTF-8 encoding
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            logger.warning(f"UTF-8 decoding failed for {file_path.name}, trying latin-1...")
            content = file_path.read_text(encoding="latin-1")
        
        # First, extract the chapter title to use as fallback category
        chapter_title = extract_chapter_title(content)
        logger.info(f"Chapter title: {chapter_title or 'Unknown'}")
        
        # Pre-scan for all diagnosis codes in the document
        # This helps with finding codes that might be mentioned in text but not in headers
        all_codes_in_doc = set(re.findall(r'\b([A-Z][0-9]{2}(?:\.[0-9]+)?)\b', content))
        logger.debug(f"Found {len(all_codes_in_doc)} potential ICD codes in document")
        
        # Find all potential diagnosis headers at different header levels (## to ####)
        diagnosis_headers = find_potential_diagnosis_headers(content)
        logger.info(f"Found {len(diagnosis_headers)} potential diagnosis headers in {file_path.name}")
        
        # Split the file into diagnosis sections
        diagnosis_sections = split_markdown_by_diagnosis(content, file_path.name)
        logger.info(f"Found {len(diagnosis_sections)} diagnosis sections in {file_path.name}")
        
        # Track processed diagnosis codes to avoid duplicates
        processed_codes = set()
        
        # First, process diagnosis headers (they often contain the most accurate code-title pairs)
        for header in diagnosis_headers:
            if any(term in header["title"].lower() for term in ["bestimme", "beachte", "codiere", "zusatzcodierung"]):
                continue
                
            # Try to extract a code from the header
            header_code = extract_code_from_text(header["title"])
            if header_code and header_code not in processed_codes:
                # Check if this code exists in our diagnoses dictionary
                if header_code in diagnoses:
                    entry = diagnoses[header_code]
                    # Update the category if needed
                    if not entry.category and chapter_title:
                        entry.category = chapter_title
                    
                    logger.info(f"Found matching diagnosis in header: {header_code} - {entry.title}")
                else:
                    # Create a new entry for this diagnosis
                    clean_title = re.sub(r'[\*\#\(\)]', '', header["title"]).strip()
                    # Remove the code from the title if it's there
                    clean_title = re.sub(r'\b{}\b'.format(header_code), '', clean_title).strip()
                    
                    new_entry = DiagnosisEntry(
                        code=header_code,
                        title=clean_title,
                        category=chapter_title,
                        source_document=file_path.name
                    )
                    diagnoses[header_code] = new_entry
                    logger.info(f"Created new diagnosis from header: {header_code} - {clean_title}")
                    
                processed_codes.add(header_code)

        # Process each diagnosis section
        for section in diagnosis_sections:
            # Skip sections that are too small or likely not diagnoses
            if not section.is_valid_diagnosis():
                continue
                
            # Determine the type of content in this section
            section_type = determine_section_type(section.content)
            logger.debug(f"Section '{section.title}' determined to be of type: {section_type}")

            # Skip sections with titles that are clearly not diagnoses
            if section.title and any(term in section.title.lower() for term in
                                    ["bestimme", "beachte", "codiere", "zusatzcodierung"]):
                continue

            # Use chapter title as fallback category if section has none
            if not section.category and chapter_title:
                section.category = chapter_title

            # Try to find the code in the section
            if section.code:
                # Clean the code - sometimes it might have extra characters
                clean_code = re.sub(r'[^A-Z0-9\.]', '', section.code)

                # Check if this code exists in our diagnoses dictionary
                codes_to_check = [clean_code, section.code]

                # Also check similar codes with different variations
                if '.' in section.code:
                    base_code = section.code.split('.')[0]
                    codes_to_check.append(base_code)
                    # Also try with a .0 suffix if it's just a base code
                    codes_to_check.append(f"{base_code}.0")
            else:
                # Try to extract a code from the title
                potential_code = extract_code_from_text(section.title)
                codes_to_check = [potential_code] if potential_code else []

            # Try to match by code first
            matched_entry = None
            for code in codes_to_check:
                if code and code in diagnoses:
                    matched_entry = diagnoses[code]
                    break

            # If no match by code, try to match by title
            if not matched_entry and section.title:
                clean_title = section.title.lower()
                # Sort existing entries by title length (descending) to prioritize more specific matches
                sorted_entries = sorted(diagnoses.values(), key=lambda e: len(e.title), reverse=True)

                for entry in sorted_entries:
                    # Check for significant overlap between titles
                    if (clean_title in entry.title.lower() or
                        entry.title.lower() in clean_title or
                        similarity_score(clean_title, entry.title.lower()) > 0.7):
                        matched_entry = entry
                        break

            # If we found a match, use appropriate processing based on section type
            if matched_entry and matched_entry.code not in processed_codes:
                logger.info(f"Found matching diagnosis for enrichment: {matched_entry.code} - {matched_entry.title} ({section_type})")
                
                # Use different extraction strategies based on the section type
                if section_type == "criteria" and not matched_entry.criteria_text:
                    # Extract diagnostic criteria
                    matched_entry.criteria_text = section.content
                    # Extract short description if none exists
                    if not matched_entry.description_short:
                        paragraphs = section.content.split('\n\n')
                        if paragraphs:
                            first_paragraph = paragraphs[0].strip()
                            matched_entry.description_short = first_paragraph[:150] + "..." if len(first_paragraph) > 150 else first_paragraph
                
                elif section_type == "features" and not matched_entry.diagnostic_features_text:
                    # Extract diagnostic features
                    matched_entry.diagnostic_features_text = section.content
                    # Extract symptom keywords if none exist
                    if not matched_entry.symptoms_keywords or len(matched_entry.symptoms_keywords) == 0:
                        extract_symptom_keywords(section.content, matched_entry)
                        
                elif section_type == "differential" and not matched_entry.differential_diagnosis_text:
                    # Extract differential diagnosis information
                    matched_entry.differential_diagnosis_text = section.content
                    
                else:
                    # For other section types, use the LLM for full extraction
                    enrich_diagnosis_with_llm(section, matched_entry, file_path)
                    
                processed_codes.add(matched_entry.code)
            elif section.code and section.title and section.code not in processed_codes:
                # If no match, this might be a new diagnosis not in the overview
                logger.info(f"Found new diagnosis not in overview: {section.code} - {section.title}")
                new_entry = DiagnosisEntry(
                    code=section.code,
                    title=section.title,
                    category=section.category or chapter_title,
                    source_document=file_path.name
                )
                diagnoses[section.code] = new_entry
                enrich_diagnosis_with_llm(section, new_entry, file_path)
                processed_codes.add(section.code)

    except Exception as e:
        logger.error(f"Error processing chapter file {file_path.name}: {str(e)}")

def extract_chapter_title(content: str) -> Optional[str]:
    """Extract the title of the chapter from the content.
    
    Attempts to find the first major heading in the content.
    """
    # Look for the first heading line
    lines = content.splitlines()
    
    # Pattern to match headings with # or ** markers
    heading_pattern = re.compile(r'^#+\s+(?:\*\*)?([^\*]+?)(?:\*\*)?\s*$')
    
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'\*\*|\*', '', title)
            return title
            
    return None

def find_potential_diagnosis_headers(content: str) -> List[Dict[str, str]]:
    """Find all potential diagnosis headers in the content.
    
    Returns a list of dictionaries with 'title' and 'level' for each potential diagnosis header.
    """
    headers = []
    lines = content.splitlines()
    
    # Patterns to match header lines that might contain diagnosis titles
    header_pattern = re.compile(r'^(#{2,4})\s+(.+?)\s*$')
    code_pattern = re.compile(r'\b([A-Z][0-9]{2}(?:\.[0-9]+)?)\b')
    
    for i, line in enumerate(lines):
        match = header_pattern.match(line)
        if match:
            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            
            # Clean up the title (remove markdown formatting)
            clean_title = re.sub(r'\*\*|\*', '', title)
            
            # Check if this appears to be a diagnosis header
            # It's a diagnosis header if it has a code or contains diagnosis-related terms
            has_code = code_pattern.search(clean_title) is not None
            diagnosis_terms = ['störung', 'syndrom', 'episode', 'diagnos', 'kriterien']
            is_diagnosis_term = any(term in clean_title.lower() for term in diagnosis_terms)
            
            # Skip very short titles that aren't likely diagnoses
            if len(clean_title) < 5:
                continue
                
            # Skip headers that are clearly not diagnoses
            if any(term in clean_title.lower() for term in 
                   ['beachte:', 'bestimme:', 'codierung', 'zusatz']):
                continue
            
            if has_code or is_diagnosis_term:
                # Look ahead a few lines to see if there's a code nearby
                if not has_code:
                    for j in range(i+1, min(i+5, len(lines))):
                        code_match = code_pattern.search(lines[j])
                        if code_match:
                            # Add the code to the title
                            clean_title += f" {code_match.group(1)}"
                            has_code = True
                            break
                
                headers.append({"title": clean_title, "level": level})
    
    return headers

def similarity_score(text1: str, text2: str) -> float:
    """Calculate a simple similarity score between two strings.

    Returns a value between 0 (no similarity) and 1 (identical).
    """
    # Normalize and tokenize texts
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0.0

def extract_code_from_text(text: str) -> Optional[str]:
    """Attempt to extract an ICD-10 code from a text."""
    if not text:
        return None
        
    # Pattern to find ICD-10 codes in text
    code_pattern = re.compile(r'\b([A-Z][0-9]{2}(?:\.[0-9]+)?)\b')
    match = code_pattern.search(text)
    
    return match.group(1) if match else None

def determine_section_type(content: str) -> str:
    """Determine the type of section based on content analysis.
    
    Returns one of: "criteria", "features", "differential", "general"
    """
    content_lower = content.lower()
    
    # Check for diagnostic criteria section
    if "diagnostische kriterien" in content_lower:
        return "criteria"
    
    # Check for sections with criteria patterns (A., B., C., etc.)
    if re.search(r'\b[A-E]\s*\.(\s+[A-Za-z]|\s+[0-9])', content_lower):
        return "criteria"
    
    # Look for numbered lists that might be criteria
    if re.search(r'\n\s*[0-9]+\.\s', content_lower):
        return "criteria"
        
    # Check for diagnostic features
    if "diagnostische merkmale" in content_lower:
        return "features"
    
    # Check for belonging features
    if "zugehörige merkmale" in content_lower:
        return "features"
    
    # Check for description-related terms
    if any(term in content_lower for term in ["prävalenz", "risiko", "verlauf", "entwicklung"]):
        return "features"
    
    # Check for differential diagnosis
    if "differenzialdiagnose" in content_lower:
        return "differential"
    
    # Check for comorbidity
    if "komorbidität" in content_lower:
        return "differential"
    
    # Default - if the content is substantial, assume it's features
    if len(content.strip()) > 500:
        return "features"
        
    # Default fallback
    return "general"

def extract_symptom_keywords(content: str, entry: DiagnosisEntry) -> None:
    """Extract symptom keywords from content text."""
    # Use regex to find symptom patterns and key phrases
    keywords = []
    
    # Common symptom patterns in German
    patterns = [
        r'(?:verlust|mangel)\s+(?:an|von)?\s+([^.,;:\n]{3,40})',
        r'(?:gesteigerte|verminderte|erhu00f6hte)\s+([^.,;:\n]{3,40})',
        r'(?:schwierigkeiten|probleme)\s+(?:mit|bei)?\s+([^.,;:\n]{3,40})',
        r'(?:gefu00fchle?\s+(?:von|der))\s+([^.,;:\n]{3,40})',
        r'\b(?:angst|furcht)\s+(?:vor)?\s+([^.,;:\n]{3,40})'
    ]
    
    # Extract keywords using patterns
    for pattern in patterns:
        matches = re.findall(pattern, content.lower())
        for match in matches:
            keyword = match.strip()
            if len(keyword) > 3 and keyword not in keywords:
                keywords.append(keyword)
    
    # Also look for bullet points or numbered lists which often contain symptoms
    bullet_points = re.findall(r'[-\*]\s+([^\n]+)', content)
    for point in bullet_points:
        # Only include shorter bullet points as they're more likely to be symptoms
        if 5 < len(point) < 100 and not any(exclude in point.lower() for exclude in ['beispiel', 'beachte']):
            keywords.append(point.strip())
    
    # Add extracted keywords to the entry
    if keywords:
        if not entry.symptoms_keywords:
            entry.symptoms_keywords = []
        entry.symptoms_keywords.extend(keywords)
        # Remove duplicates
        entry.symptoms_keywords = list(set(entry.symptoms_keywords))

def enrich_diagnosis_with_llm(section: MarkdownSection, entry: DiagnosisEntry, file_path: Path) -> None:
    """Use the LLM to extract detailed information for a diagnosis entry."""
    # Skip if content is too short
    if len(section.content.strip()) < 50:
        logger.warning(f"  Section '{section.title}' has too little content to process with LLM")
        return
    
    # Create a prompt with additional context
    prompt_content = (
        f"Source Filename: {file_path.name}\n\n"
        f"Diagnosis Code: {entry.code or 'Unknown'}\n"
        f"Diagnosis Title: {entry.title}\n"
        f"Category: {section.category or 'Unknown'}\n\n"
        f"---\n\n"
        f"{section.content}\n\n"
        f"---\n\n"
        f"Instructions: Extract structured information about this diagnosis according to the DiagnosisOutput model. " 
        f"Pay special attention to symptoms, diagnostic criteria, and a concise description. " 
        f"If a specific field is not mentioned in the content, leave it empty."
    )

    try:
        # Extract structured data using the LLM agent
        result = extraction_agent.run_sync(prompt_content)
        extracted_data = result.output

        # Update the entry with the extracted data, preserving the original code and title
        original_code = entry.code
        original_title = entry.title

        # Convert to dict and update our entry
        extracted_dict = extracted_data.model_dump()
        entry.update_from_dict(extracted_dict)

        # Preserve the original code and title from the overview
        entry.code = original_code

        # Only override title if the extracted one is significantly longer and might contain more information
        if extracted_data.title and len(extracted_data.title) > len(original_title) * 1.5:
            logger.info(f"Replacing title '{original_title}' with longer extracted title '{extracted_data.title}'")
            entry.title = extracted_data.title
        else:
            entry.title = original_title

        logger.info(f"Successfully enriched diagnosis: {entry.code} - {entry.title}")

    except Exception as e:
        logger.error(f"Error extracting details for {entry.code} - {entry.title}: {str(e)}")

def enrich_diagnoses_from_detailed_files(diagnoses: Dict[str, DiagnosisEntry]) -> Dict[str, DiagnosisEntry]:
    """Enrich diagnosis entries with details from chapter files."""
    chapter_files = list(DIAGNOSES_DIR.glob("*.md"))
    
    if not chapter_files:
        logger.warning(f"No chapter files found in {DIAGNOSES_DIR}")
        return diagnoses
    
    logger.info(f"Found {len(chapter_files)} chapter file(s) to process for enrichment")
    
    for chapter_file in chapter_files:
        logger.info(f"Processing chapter file: {chapter_file.name}")
        try:
            process_chapter_file(chapter_file, diagnoses)
        except Exception as e:
            logger.error(f"Error processing chapter file {chapter_file.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Count how many diagnoses have been enriched with descriptions
    enriched_count = sum(1 for entry in diagnoses.values() if entry.description_short is not None)
    criteria_count = sum(1 for entry in diagnoses.values() if entry.criteria_text is not None)
    features_count = sum(1 for entry in diagnoses.values() if entry.diagnostic_features_text is not None)
    
    logger.info(f"Enrichment statistics:")
    logger.info(f"  - {enriched_count} of {len(diagnoses)} entries have descriptions")
    logger.info(f"  - {criteria_count} of {len(diagnoses)} entries have diagnostic criteria")
    logger.info(f"  - {features_count} of {len(diagnoses)} entries have diagnostic features")
    
    return diagnoses

def main():
    """Main function to process markdown files and generate JSON."""
    logger.info("Starting diagnosis extraction process...")
    
    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract diagnoses from overview files
    diagnoses_dict = extract_diagnoses_from_overview()
    logger.info(f"Extracted {len(diagnoses_dict)} diagnoses from overview files")
    
    # Save the preliminary diagnoses to cache
    save_diagnoses_cache(diagnoses_dict)
    
    # Step 2: Enrich diagnoses with details from chapter files
    enriched_diagnoses = enrich_diagnoses_from_detailed_files(diagnoses_dict)
    
    # Step 2b: Process specific files that are known to contain rich content
    # This ensures we process important files even if other parts of the script miss them
    important_files = ["dsm-5_2_03.md"]  # Add more key files as needed
    for filename in important_files:
        file_path = DIAGNOSES_DIR / filename
        if file_path.exists():
            logger.info(f"Ensuring processing of important file: {filename}")
            process_chapter_file(file_path, enriched_diagnoses)
    
    # Step 3: Convert the diagnoses dictionary to a list for output
    all_diagnoses = [entry.to_dict() for entry in enriched_diagnoses.values()]
    
    # Step 4: Validate and save results
    validate_and_save_output(all_diagnoses)
    
    # Summarize results
    if all_diagnoses:
        logger.info(f"Process complete. Extracted and enriched {len(all_diagnoses)} diagnoses.")
    else:
        logger.warning("Process complete. No diagnoses were successfully extracted.")


if __name__ == "__main__":
    # Note: pydantic-ai's run_sync uses asyncio internally.
    # Depending on your environment (e.g., Jupyter), you might need
    # to manage the asyncio event loop explicitly if you encounter issues.
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise
