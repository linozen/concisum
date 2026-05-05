from pathlib import Path
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
        # Handle multiple JSON formats
        if "ref_text" in utt and "ref_spk" in utt:
            # Format: utterance .json from 'diarizationlm'
            text = utt["ref_text"]
            speaker = utt["ref_spk"].split()[0]  # Take the first speaker ID
            utterances.append(Utterance(text=text, speaker=speaker))
        elif "ref_text" in utt and "speaker" in utt:
            # Format: ground truth .utt.json (ref_text + numeric speaker)
            text = utt["ref_text"]
            speaker = str(utt["speaker"])
            utterances.append(Utterance(text=text, speaker=speaker))
        elif "text" in utt and "speaker" in utt:
            # Format: regular whisper-generated .json file from verbatim
            text = utt["text"]
            speaker = utt["speaker"][-1]  # Extract the last character
            utterances.append(Utterance(text=text, speaker=speaker))

    return UtteranceList(utterances=utterances)


def parse_utterance_list_from_dict(data: dict) -> UtteranceList:  # type: ignore[type-arg]
    """Parse a dict (same schema as the JSON files) into an UtteranceList."""
    utterances = []
    for utt in data.get("utterances", []):
        if "ref_text" in utt and "ref_spk" in utt:
            utterances.append(Utterance(text=utt["ref_text"], speaker=utt["ref_spk"].split()[0]))
        elif "ref_text" in utt and "speaker" in utt:
            utterances.append(Utterance(text=utt["ref_text"], speaker=str(utt["speaker"])))
        elif "text" in utt and "speaker" in utt:
            utterances.append(Utterance(text=utt["text"], speaker=str(utt["speaker"])))
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
