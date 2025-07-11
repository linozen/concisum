import json
import re
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent

"""
This script extracts diagnoses information from DSM-5 and ICD-10 formatted markdown files.
It uses a simple chunking strategy to process large files and lets the LLM identify and extract
diagnoses information according to a defined schema.

The script processes both overview files for basic diagnosis information and detailed chapter files
for enrichment, then merges the results into a comprehensive, potentially nested JSON output
structured primarily around ICD-10 codes.
"""

# --- Configuration ---
OVERVIEW_DIR = Path("data/refined/0_overview")  # Directory containing overview files with codes and titles
DIAGNOSES_DIR = Path("data/refined/2_diagnoses")  # Directory containing detailed diagnosis chapters
OUTPUT_FILE = Path("data/diagnoses_extracted_hierarchical.json") # Updated output file name
LOG_FILE = Path("logs/extraction_hierarchical.log") # Updated log file name
LOG_LEVEL = logging.INFO
DIAGNOSES_CACHE_FILE = Path("data/diagnosis_cache_hierarchical.json") # Updated cache file name

# LLM model to use
LLM_MODEL = "google-gla:gemini-2.0-flash"

# Maximum chunk size in characters
MAX_CHUNK_SIZE = 15000 # Keep chunk size manageable for the LLM

# Configure logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("diagnosis_extractor_hierarchical")

# --- Pydantic Model for LLM Extraction Output ---

class DiagnosisChunkOutput(BaseModel):
    """Pydantic model defining the structured output extracted by the LLM *from a chunk*.
    This represents a single diagnosis or subtype found within the chunk.
    The merging logic will assemble the final hierarchical structure.
    """
    icd_10_code: Optional[str] = Field(None, description="Der primäre ICD-10-Code (z.B. 'F31.8', 'F34.0', 'F31.81'), der in diesem Textabschnitt identifiziert wurde. Konzentriere dich auf Codes mit Punkt und mindestens einer Ziffer danach.")
    parent_icd_10_code: Optional[str] = Field(None, description="Falls dieser Code ein spezifischer Subtyp ist (z.B. 'F31.81'), gib hier den übergeordneten ICD-10-Code an (z.B. 'F31.8'), falls im Text erkennbar.")
    icd_10_title: Optional[str] = Field(None, description="Der offizielle Titel für den 'icd_10_code' gemäß ICD-10, falls im Text vorhanden.")
    icd_10_definition: Optional[str] = Field(None, description="Die Definition oder Beschreibung für den 'icd_10_code' gemäß ICD-10 (oft unter 'Definition' oder 'Inkl.:'), falls im Text vorhanden.")
    dsm_5_title: Optional[str] = Field(None, description="Der spezifische DSM-5-Titel für die Störung, die diesem Code entspricht (z.B. 'Bipolar-II-Störung' für F31.81), falls im Text erwähnt.")
    dsm_5_category: Optional[str] = Field(None, description="Die übergeordnete DSM-5-Kategorie (z.B. 'Bipolare und verwandte Störungen'), extrahiert aus H1/H2-Überschriften oder dem Kontext.")
    # Keep detailed fields, as they might apply to either main code or subtype depending on context
    description_short: Optional[str] = Field(None, description="Eine kurze, prägnante Zusammenfassung (1-2 Sätze) der Störung/des Subtyps aus dem Einleitungstext.")
    symptoms_keywords: List[str] = Field(default_factory=list, description="Eine Liste spezifischer, prägnanter deutscher Schlüsselwörter/Phrasen für Symptome/Merkmale.")
    criteria_text: Optional[str] = Field(None, description="Vollständiger Text der diagnostischen Kriterien (z.B. Abschnitte A, B, C).")
    diagnostic_features_text: Optional[str] = Field(None, description="Detaillierter Text über diagnostische oder zugehörige Merkmale.")
    severity_levels_text: Optional[str] = Field(None, description="Text über Schweregrade (leicht, mittel, schwer).") # Renamed for clarity
    differential_diagnosis_text: Optional[str] = Field(None, description="Text über Differentialdiagnosen.")
    comorbidity_text: Optional[str] = Field(None, description="Text über Komorbiditäten.")
    specifiers_text: Optional[str] = Field(None, description="Text, der spezifische Zusatzkodierungen oder Spezifizierer beschreibt (z.B. 'Mit Angst', 'Mit Rapid Cycling').")
    coding_notes_text: Optional[str] = Field(None, description="Text mit spezifischen Kodierungshinweisen oder -konventionen.")
    examples_text: Optional[str] = Field(None, description="Falls es sich um eine 'Andere Näher Bezeichnete' Kategorie handelt, extrahiere die Beispiele (z.B. für F31.89).")
    source_document: Optional[str] = Field(None, description="Der Pfad zur Quell-Markdown-Datei (wird nach der Extraktion hinzugefügt).") # Added later


# --- Pydantic AI Agent Definition ---

extraction_agent = Agent(
    LLM_MODEL,
    output_type=List[DiagnosisChunkOutput], # Expecting a list of diagnoses/subtypes found in the chunk
    instructions="""
    You are an expert medical information extractor specializing in psychiatric diagnostic manuals like DSM-5 and ICD-10.
    You will be given a chunk of Markdown text containing information about one or more psychiatric diagnoses or subtypes.

    Your task is to extract ALL diagnoses or subtypes found in the text chunk and output them as a list of DiagnosisChunkOutput objects.
    For each diagnosis/subtype, fill in as many fields as possible based ONLY on the information present in THIS CHUNK.

    Key Focus Areas:
    1.  **Identify Codes:** Extract the primary ICD-10 code (e.g., F31.8, F34.0, F31.81). Look for patterns like F##.# or F##.##.
    2.  **Identify Parent Code:** If the identified code is clearly a subtype (like F31.81), determine its parent code (like F31.8) if context allows.
    3.  **Extract Titles:** Capture both ICD-10 and DSM-5 titles if available for the specific code.
    4.  **Capture Definitions:** Extract ICD-10 definitions if present.
    5.  **Extract DSM Details:** Find the DSM-5 category, create a short description, list symptom keywords (German), and extract full text for criteria, features, severity, differential diagnosis, comorbidity, specifiers, coding notes, and examples (especially for 'Other Specified' types like F31.89).
    6.  **German Language:** Use German terminology and phrasing as found in the source text for keywords and descriptions.

    Important Rules:
    *   If a field's information isn't in the current chunk, leave it as null or empty list. Do NOT invent information.
    *   If multiple distinct diagnoses or subtypes are discussed (e.g., Bipolar I criteria followed by Bipolar II criteria), create separate DiagnosisChunkOutput objects for each.
    *   Focus on extracting information directly associated with a specific identified code within the chunk.
    *   If the text seems incomplete (continuation from a previous chunk), extract what you can for the codes mentioned.
    """
)

# --- Dataclass for Final Merged Structure ---

@dataclass
class FinalDiagnosisEntry:
    """Represents a final, potentially nested diagnosis entry centered around an ICD-10 code."""
    icd_10_code: str
    icd_10_title: Optional[str] = None
    icd_10_definition: Optional[str] = None
    icd_10_source: Optional[str] = "ICD-10-GM 2025" # Default or extracted

    # DSM-5 specific fields (populated if direct equivalent or via subtypes)
    dsm_5_mapping: Dict[str, Any] = field(default_factory=dict) # Stores mapping type and notes
    dsm_5_criteria_text: Optional[str] = None # For direct equivalents like F34.0
    dsm_5_specifiers_text: Optional[str] = None # For direct equivalents or potentially main category
    dsm_5_coding_notes_text: Optional[str] = None # For direct equivalents or potentially main category
    dsm_5_diagnostic_features_text: Optional[str] = None # For direct equivalents
    dsm_5_severity_levels_text: Optional[str] = None
    dsm_5_differential_diagnosis_text: Optional[str] = None
    dsm_5_comorbidity_text: Optional[str] = None
    dsm_5_examples_text: Optional[str] = None # e.g. for F31.89 details

    # List to hold nested subtypes (as dictionaries matching the final output structure for a subtype)
    dsm_5_subtypes: List[Dict[str, Any]] = field(default_factory=list)

    # Keep track of source files contributing to this entry
    source_documents: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the entry to a dictionary for JSON serialization, cleaning up."""
        data = {
            "icd-10-code": self.icd_10_code,
            "icd-10-title": self.icd_10_title,
            "icd-10-definition": self.icd_10_definition,
            "icd-10-source": self.icd_10_source,
            "dsm-5-mapping": self.dsm_5_mapping,
            # Include DSM-5 fields only if they have content
            **{k.replace('_text', '').replace('dsm_5_','dsm-5-'): v for k, v in self.__dict__.items()
               if k.startswith('dsm_5_') and k.endswith('_text') and v is not None},
            "dsm-5-subtypes": self.dsm_5_subtypes,
            "source_documents": sorted(list(set(self.source_documents))) # Unique, sorted list
        }
        # Remove keys with None values or empty lists/dicts (except subtypes)
        return {k: v for k, v in data.items() if v or k == 'dsm-5-subtypes'}


    def update_field(self, field_name: str, value: Optional[Any], source_doc: str):
        """Update a field only if the new value is not None and potentially richer."""
        if value is not None:
            current_value = getattr(self, field_name, None)
            # Simple overwrite for now, could add logic to merge lists or prefer longer text
            if value != current_value:
                 # Basic check: prefer longer text descriptions
                if isinstance(value, str) and isinstance(current_value, str) and len(value) < len(current_value):
                    pass # Keep the longer existing text
                else:
                    setattr(self, field_name, value)
        if source_doc and source_doc not in self.source_documents:
            self.source_documents.append(source_doc)


    def update_subtype(self, subtype_data: Dict[str, Any], source_doc: str):
        """Add or update a subtype in the dsm_5_subtypes list."""
        subtype_code = subtype_data.get("icd_10_code")
        if not subtype_code:
            logger.warning(f"Attempted to update subtype without code for parent {self.icd_10_code}")
            return

        found = False
        for i, existing_subtype in enumerate(self.dsm_5_subtypes):
            if existing_subtype.get("icd-10-code") == subtype_code:
                # Update existing subtype - simplistic merge, favoring new non-null values
                for key, value in subtype_data.items():
                     # Map DiagnosisChunkOutput keys to FinalDiagnosisEntry keys/structure
                    mapped_key = key.replace('_text', '').replace('dsm_5_','dsm-5-')
                    if mapped_key.startswith("icd_10"): mapped_key = mapped_key.replace("icd_10", "icd-10")

                    if value is not None and value != []:
                         # Prefer longer text descriptions for subtypes too
                        current_sub_value = existing_subtype.get(mapped_key)
                        if isinstance(value, str) and isinstance(current_sub_value, str) and len(value) < len(current_sub_value):
                             continue # Keep existing longer text
                        existing_subtype[mapped_key] = value
                found = True
                break

        if not found:
             # Convert to final format before adding
            final_subtype_dict = {
                "icd-10-code": subtype_data.get("icd_10_code"),
                "dsm-5-title": subtype_data.get("dsm_5_title"),
                "dsm-5-category": subtype_data.get("dsm_5_category"),
                 # Include DSM-5 fields only if they have content
                **{k.replace('_text', '').replace('dsm_5_','dsm-5-'): v for k, v in subtype_data.items()
                   if k.startswith('dsm_5_') and k.endswith('_text') and v is not None},
                "description_short": subtype_data.get("description_short"),
                "symptoms_keywords": subtype_data.get("symptoms_keywords", []),
                # Add other relevant fields extracted by DiagnosisChunkOutput
                "criteria": subtype_data.get("criteria_text"),
                "diagnostic-features": subtype_data.get("diagnostic_features_text"),
                "severity-levels": subtype_data.get("severity_levels_text"),
                "differential-diagnosis": subtype_data.get("differential_diagnosis_text"),
                "comorbidity": subtype_data.get("comorbidity_text"),
                "specifiers": subtype_data.get("specifiers_text"),
                "coding-notes": subtype_data.get("coding_notes_text"),
                "examples": subtype_data.get("examples_text"),
            }
            # Remove keys with None values or empty lists
            final_subtype_dict = {k: v for k, v in final_subtype_dict.items() if v or v == [] and k == 'symptoms_keywords'}
            self.dsm_5_subtypes.append(final_subtype_dict)

        if source_doc and source_doc not in self.source_documents:
            self.source_documents.append(source_doc)


# --- Helper Functions ---

def extract_chapter_title_simple(content: str) -> Optional[str]:
    """Extract the first H1 or H2 title from the content."""
    lines = content.splitlines()
     # Match H1 or H2, capture content, remove markdown bold/italics
    heading_pattern = re.compile(r'^#{1,2}\s+(?:\*\*?)?([^#\*].*?)(?:\*\*?)?\s*$')
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            return match.group(1).strip()
    return None

def split_large_content(content: str) -> List[str]:
    """Split large content into smaller chunks, trying to keep sections together."""
    chunks = []
    lines = content.splitlines()
    current_chunk_lines = []
    current_size = 0
    # Regex to identify potential section breaks (H2 or H3 typically denote new diagnoses/subsections)
    section_break_pattern = re.compile(r'^#{2,3}\s+')

    for line in lines:
        line_size = len(line.encode('utf-8')) # Use byte length for more accuracy

        # Check if adding this line exceeds the limit
        if current_size + line_size > MAX_CHUNK_SIZE and current_chunk_lines:
            # Check if the current line is a section break
            is_section_break = section_break_pattern.match(line)

            # If the current line IS NOT a section break, try to find the *last* section break
            # in the current chunk to split there, keeping sections less fragmented.
            split_index = -1
            if not is_section_break:
                for i in range(len(current_chunk_lines) - 1, max(-1, len(current_chunk_lines) - 10), -1): # Look back ~10 lines
                     if section_break_pattern.match(current_chunk_lines[i]):
                        split_index = i
                        break

            if split_index != -1:
                 # Split at the last found section break
                chunks.append("\n".join(current_chunk_lines[:split_index]))
                current_chunk_lines = current_chunk_lines[split_index:]
                current_size = sum(len(l.encode('utf-8')) for l in current_chunk_lines)
            else:
                 # If no recent section break or the current line is a break, split here
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_size = 0

        # Add the line to the current chunk
        current_chunk_lines.append(line)
        current_size += line_size

    # Add the final chunk
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    logger.debug(f"Split content into {len(chunks)} chunks.")
    return chunks

def extract_diagnoses_from_file(file_path: Path) -> List[Dict]:
    """Process a single markdown file, chunk it, and extract diagnoses using LLM."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []

    chapter_title = extract_chapter_title_simple(content)
    logger.info(f"Processing file: {file_path.name}, Chapter Context: {chapter_title or 'Unknown'}")

    chunks = split_large_content(content)
    logger.info(f"Split {file_path.name} into {len(chunks)} chunks.")

    extracted_data_list = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} of {file_path.name}")

        # Add context to the prompt
        prompt = f"""
        Gesamtdokument: {file_path.name}
        Übergeordnetes Kapitel (Kontext): {chapter_title or 'Unbekannt'}
        Aktueller Textabschnitt (Chunk {i+1}/{len(chunks)}):

        --- TEXT START ---
        {chunk}
        --- TEXT END ---

        Extrahiere ALLE Diagnosen oder Subtypen aus diesem Textabschnitt gemäß dem DiagnosisChunkOutput Schema.
        Gib eine Liste von Objekten zurück. Fülle nur Felder aus, deren Informationen DIREKT in DIESEM Textabschnitt stehen.
        Achte auf ICD-10 Codes (z.B. Fxx.y, Fxx.yz) und deren mögliche Eltern-Codes (z.B. Fxx.y ist Elternteil von Fxx.yz).
        Extrahiere deutsche Symptom-Schlüsselwörter.
        """

        try:
            # Explicitly use run_sync for synchronous execution if needed
            result = extraction_agent.run_sync(user_prompt=prompt) # Pass prompt explicitly

            if result and result.output:
                logger.info(f"LLM Extracted {len(result.output)} items from chunk {i+1}")
                for diagnosis_output in result.output:
                    # Convert Pydantic model to dict and add source document
                    diagnosis_dict = diagnosis_output.model_dump()
                    diagnosis_dict["source_document"] = str(file_path.name) # Add source doc here
                    extracted_data_list.append(diagnosis_dict)
            else:
                logger.warning(f"No diagnoses extracted by LLM from chunk {i+1} of {file_path.name}")

        except Exception as e:
            logger.error(f"Error processing chunk {i+1} of {file_path.name} with LLM: {str(e)}", exc_info=True) # Log stack trace

    logger.info(f"Finished processing {file_path.name}, extracted {len(extracted_data_list)} raw items.")
    return extracted_data_list


def merge_diagnoses(existing_diagnoses: Dict[str, FinalDiagnosisEntry], new_data_list: List[Dict]) -> Dict[str, FinalDiagnosisEntry]:
    """Merge new extracted data into the existing diagnoses dictionary, handling nesting."""
    for data in new_data_list:
        icd_code = data.get("icd_10_code")
        parent_code = data.get("parent_icd_10_code")
        source_doc = data.get("source_document", "unknown")

        if not icd_code:
            logger.warning(f"Skipping entry without ICD-10 code: {data.get('dsm_5_title') or data.get('icd_10_title')}")
            continue

        # Determine if this is a subtype entry based on parent_code or code format (e.g., Fxx.yz vs Fxx.y)
        is_subtype = bool(parent_code)
        if not is_subtype and '.' in icd_code and len(icd_code.split('.'))>1 and len(icd_code.split('.')[1]) > 1 : # Heuristic: Fxx.yz is likely subtype of Fxx.y
            potential_parent = icd_code.split('.')[0] + '.' + icd_code.split('.')[1][0]
            # Check if the potential parent exists or if the code itself might be the parent
            if potential_parent != icd_code:
                 parent_code = potential_parent
                 is_subtype = True
                 logger.debug(f"Inferred parent code {parent_code} for subtype {icd_code}")

        if is_subtype and parent_code:
            # This is a subtype, find or create the parent
            if parent_code not in existing_diagnoses:
                logger.info(f"Creating placeholder parent entry for {parent_code} based on subtype {icd_code}")
                existing_diagnoses[parent_code] = FinalDiagnosisEntry(icd_10_code=parent_code)
                # Attempt to add parent title/definition if subtype provided it (unlikely but possible)
                if data.get("icd_10_title") and parent_code in data.get("icd_10_title", ""): # Basic check
                     existing_diagnoses[parent_code].icd_10_title = data.get("icd_10_title")
                if data.get("icd_10_definition") and parent_code in data.get("icd_10_definition", ""):
                     existing_diagnoses[parent_code].icd_10_definition = data.get("icd_10_definition")

            logger.debug(f"Updating/adding subtype {icd_code} under parent {parent_code}")
            existing_diagnoses[parent_code].update_subtype(data, source_doc)

        else:
            # This is a main entry (or a subtype whose parent wasn't identified)
            if icd_code not in existing_diagnoses:
                logger.info(f"Creating new main entry for {icd_code}: {data.get('icd_10_title') or data.get('dsm_5_title')}")
                existing_diagnoses[icd_code] = FinalDiagnosisEntry(icd_10_code=icd_code)

            logger.debug(f"Updating main entry for {icd_code}")
            entry = existing_diagnoses[icd_code]
            entry.update_field("icd_10_title", data.get("icd_10_title"), source_doc)
            entry.update_field("icd_10_definition", data.get("icd_10_definition"), source_doc)
            # If DSM title/category provided for main code, set mapping
            if data.get("dsm_5_title"):
                 entry.dsm_5_mapping["type"] = "direct_equivalent" # Assume direct if DSM title found at main level
                 entry.dsm_5_mapping["dsm-5-title"] = data.get("dsm_5_title")
            if data.get("dsm_5_category"):
                 entry.dsm_5_mapping["dsm-5-category"] = data.get("dsm_5_category")

            # Update DSM detail fields (these belong here for direct equivalents)
            entry.update_field("dsm_5_criteria_text", data.get("criteria_text"), source_doc)
            entry.update_field("dsm_5_diagnostic_features_text", data.get("diagnostic_features_text"), source_doc)
            entry.update_field("dsm_5_severity_levels_text", data.get("severity_levels_text"), source_doc)
            entry.update_field("dsm_5_differential_diagnosis_text", data.get("differential_diagnosis_text"), source_doc)
            entry.update_field("dsm_5_comorbidity_text", data.get("comorbidity_text"), source_doc)
            entry.update_field("dsm_5_specifiers_text", data.get("specifiers_text"), source_doc)
            entry.update_field("dsm_5_coding_notes_text", data.get("coding_notes_text"), source_doc)
            entry.update_field("dsm_5_examples_text", data.get("examples_text"), source_doc)

            # Also add source document
            if source_doc and source_doc not in entry.source_documents:
                entry.source_documents.append(source_doc)

    # Post-processing: Refine DSM mapping type if subtypes exist but no direct DSM title was found
    for code, entry in existing_diagnoses.items():
        if entry.dsm_5_subtypes and not entry.dsm_5_mapping.get("dsm-5-title"):
            entry.dsm_5_mapping["type"] = "broader_category"
            entry.dsm_5_mapping["notes"] = "ICD-10 code covers multiple DSM-5 subtypes listed below."


    return existing_diagnoses

def save_diagnoses_cache(diagnoses: Dict[str, FinalDiagnosisEntry]) -> None:
    """Save the diagnoses cache to a JSON file."""
    try:
        diagnoses_dict = {code: entry.to_dict() for code, entry in diagnoses.items()}
        DIAGNOSES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DIAGNOSES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(diagnoses_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(diagnoses)} diagnoses to cache file: {DIAGNOSES_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving diagnoses cache: {str(e)}")

def load_diagnoses_cache() -> Dict[str, FinalDiagnosisEntry]:
    """Load the diagnoses cache from the JSON file."""
    diagnoses = {}
    if not DIAGNOSES_CACHE_FILE.exists():
        logger.warning(f"Diagnoses cache file not found: {DIAGNOSES_CACHE_FILE}")
        return diagnoses
    try:
        with open(DIAGNOSES_CACHE_FILE, 'r', encoding='utf-8') as f:
            diagnoses_dict = json.load(f)
        # Convert dictionaries back to FinalDiagnosisEntry objects
        for code, entry_dict in diagnoses_dict.items():
            # Need to handle reconstruction carefully if subtypes are complex objects
            # For now, assuming subtypes are stored as dicts in the cache too
            subtypes = entry_dict.pop("dsm-5-subtypes", []) # Pop subtypes before init
            sources = entry_dict.pop("source_documents", []) # Pop sources

            # Map snake_case back if necessary or adjust keys
            entry_dict_mapped = {}
            for k, v in entry_dict.items():
                 py_key = k.replace('-', '_')
                 # Map specific fields if needed, e.g.,
                 if k == "dsm-5-criteria": py_key = "dsm_5_criteria_text"
                 elif k == "dsm-5-diagnostic-features": py_key = "dsm_5_diagnostic_features_text"
                 # ... add mappings for other DSM fields ...
                 elif k == "icd-10-code": py_key = "icd_10_code"
                 elif k == "icd-10-title": py_key = "icd_10_title"
                 elif k == "icd-10-definition": py_key = "icd_10_definition"
                 elif k == "icd-10-source": py_key = "icd_10_source"
                 elif k == "dsm-5-mapping": py_key = "dsm_5_mapping"


                 entry_dict_mapped[py_key] = v


            diagnoses[code] = FinalDiagnosisEntry(**entry_dict_mapped)
            diagnoses[code].dsm_5_subtypes = subtypes # Re-assign subtypes list
            diagnoses[code].source_documents = sources # Re-assign sources list

        logger.info(f"Loaded {len(diagnoses)} diagnoses from cache file: {DIAGNOSES_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error loading diagnoses cache: {str(e)}", exc_info=True)
    return diagnoses


def validate_and_save_output(all_diagnoses_dict: Dict[str, FinalDiagnosisEntry]) -> None:
    """Validate and save the extracted diagnoses to the final JSON file."""
    if not all_diagnoses_dict:
        logger.warning("No diagnoses were successfully merged.")
        return

    # Convert final dictionary to the desired list format for output
    final_diagnoses_list = [entry.to_dict() for entry in all_diagnoses_dict.values()]

    logger.info(f"Writing {len(final_diagnoses_list)} final diagnoses entries to {OUTPUT_FILE}...")

    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "source_files": sorted(list(set(doc for entry in final_diagnoses_list for doc in entry.get("source_documents", [])))),
                "diagnosis_count": len(final_diagnoses_list),
                "extraction_model": LLM_MODEL
            },
            "diagnoses": final_diagnoses_list
        }
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully wrote final JSON output to {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"ERROR: Failed to write final output JSON file: {str(e)}", exc_info=True)


# --- Main Execution Logic ---

def process_files(file_paths: List[Path], existing_diagnoses: Dict[str, FinalDiagnosisEntry]) -> Dict[str, FinalDiagnosisEntry]:
    """Process a list of files and merge results."""
    for file_path in file_paths:
        extracted_data = extract_diagnoses_from_file(file_path)
        if extracted_data:
             existing_diagnoses = merge_diagnoses(existing_diagnoses, extracted_data)
        else:
            logger.warning(f"No data extracted from {file_path.name}")
        # Save cache incrementally after each file
        save_diagnoses_cache(existing_diagnoses)
    return existing_diagnoses

def main():
    """Main function to process markdown files and generate JSON."""
    logger.info("Starting hierarchical diagnosis extraction process...")

    # Load cache if it exists, otherwise start fresh
    diagnoses_dict = load_diagnoses_cache()
    if not diagnoses_dict:
         logger.info("No cache found or cache empty, starting fresh extraction.")

    # Determine which files need processing (basic approach: process all)
    # More sophisticated: check cache timestamps vs file modification times
    overview_files = list(OVERVIEW_DIR.glob("*.md"))
    chapter_files = list(DIAGNOSES_DIR.glob("*.md"))

    if not overview_files and not chapter_files:
        logger.error("No overview or chapter files found in specified directories. Exiting.")
        return

    # Process overview files first might help establish parent codes
    logger.info(f"--- Processing {len(overview_files)} Overview Files ---")
    diagnoses_dict = process_files(overview_files, diagnoses_dict)

    # Process detailed chapter files for enrichment and subtypes
    logger.info(f"--- Processing {len(chapter_files)} Detailed Chapter Files ---")
    diagnoses_dict = process_files(chapter_files, diagnoses_dict)

    # Final save of the potentially enriched cache
    save_diagnoses_cache(diagnoses_dict)

    # Validate and save the final output JSON
    validate_and_save_output(diagnoses_dict)

    # Summarize results
    if diagnoses_dict:
        subtype_count = sum(len(entry.dsm_5_subtypes) for entry in diagnoses_dict.values())
        logger.info(f"Process complete. Final dataset contains {len(diagnoses_dict)} top-level ICD-10 entries with {subtype_count} nested DSM-5 subtypes.")
    else:
        logger.warning("Process complete. No diagnoses were successfully extracted or merged.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}", exc_info=True)
