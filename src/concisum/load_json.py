from pathlib import Path
from typing import Dict, Any, Union
import json

from concisum.summary.models import Utterance, UtteranceList


def parse_utterance_list(file_path: str | Path) -> UtteranceList:
    """
    Parse a JSON file containing utterances into an UtteranceList object.

    Args:
        file_path: Path to the JSON file

    Returns:
        UtteranceList object containing the parsed utterances
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    utterances = []

    for utt in data.get("utterances", []):
        # Handle both JSON formats
        if "ref_text" in utt and "ref_spk" in utt:
            # Format: utterance .json from 'diarizationlm'
            text = utt["ref_text"]
            speaker = utt["ref_spk"].split()[0]  # Take the first speaker ID
            utterances.append(Utterance(text=text, speaker=speaker))
        elif "text" in utt and "speaker" in utt:
            # Format: regular whisper-generated .json file from verbatim
            text = utt["text"]
            speaker = utt["speaker"][-1]  # Extract the last character
            utterances.append(Utterance(text=text, speaker=speaker))

    return UtteranceList(utterances=utterances)


def load_utterances_from_json(file_path: str | Path) -> UtteranceList:
    """
    Simplified function to load utterances from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        UtteranceList object
    """
    return parse_utterance_list(file_path)
