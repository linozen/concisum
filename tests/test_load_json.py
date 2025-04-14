import json
import os
import pytest

from concisum.load_json import parse_utterance_list, load_utterances_from_json
from concisum.summary.models import Utterance, UtteranceList


def get_test_file_path(filename):
    """Helper function to get the absolute path to test data files"""
    return os.path.join(os.path.dirname(__file__), "data/ground_truth", filename)


class TestLoadJSON:
    def test_parse_utterance_list_diarizationlm_format(self):
        """Test parsing JSON in diarizationlm format"""
        file_path = get_test_file_path("2spk_de_AusschnittTherapieDJIMic2.json")
        result = parse_utterance_list(file_path)

        # Check result type
        assert isinstance(result, UtteranceList)

        # Check number of utterances
        assert len(result.utterances) == 33

        # Check first utterance
        assert result.utterances[0].text == "Ja, herzlich willkommen an der Ambulanz am Otto-Selz-Institut."
        assert result.utterances[0].speaker == "1"

        # Check a middle utterance
        assert result.utterances[15].text == "Tatsächlich geht es heute darum, dass wir uns beide erst einmal ein bisschen kennenlernen."
        assert result.utterances[15].speaker == "1"

        # Check last utterance
        assert result.utterances[-1].text == "Aber gut, dass sie sich die Fragen schon gestellt haben."
        assert result.utterances[-1].speaker == "1"

    def test_parse_utterance_list_whisper_format(self):
        """Test parsing JSON in whisper format"""
        file_path = get_test_file_path("2spk_de_BeltzSchwierigeSituation_1.json")
        result = parse_utterance_list(file_path)

        # Check result type
        assert isinstance(result, UtteranceList)

        # Check sample utterances
        assert result.utterances[0].text == " Hallo, Herr Kroll."
        assert result.utterances[0].speaker == "1"  # SPEAKER_01 -> "1"

        assert result.utterances[1].text == " Hallo."
        assert result.utterances[1].speaker == "0"  # SPEAKER_00 -> "0"

        # Check handling of duplicate IDs
        # There are multiple utterances with id "utt85" in the sample
        # Verify we're parsing them all correctly

        # Find all text for utterances by SPEAKER_00 or SPEAKER_01
        speaker_00_texts = [u.text for u in result.utterances if u.speaker == "0"]
        speaker_01_texts = [u.text for u in result.utterances if u.speaker == "1"]

        assert " Das tut mir leid." in speaker_00_texts
        assert " Okay, ist okay." in speaker_01_texts

    def test_load_utterances_from_json(self):
        """Test the simplified load function with both formats"""
        # Test with diarizationlm format
        file_path1 = get_test_file_path("2spk_de_AusschnittTherapieDJIMic2.json")
        result1 = load_utterances_from_json(file_path1)
        assert isinstance(result1, UtteranceList)
        assert len(result1.utterances) > 0

        # Test with whisper format
        file_path2 = get_test_file_path("2spk_de_BeltzSchwierigeSituation_1.json")
        result2 = load_utterances_from_json(file_path2)
        assert isinstance(result2, UtteranceList)
        assert len(result2.utterances) > 0

    def test_file_not_found(self):
        """Test handling of non-existent file"""
        with pytest.raises(FileNotFoundError):
            parse_utterance_list("non_existent_file.json")

    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file"""
        # Create an invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            parse_utterance_list(str(invalid_file))

    def test_empty_json(self, tmp_path):
        """Test handling of empty JSON file"""
        # Create an empty JSON file
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")

        result = parse_utterance_list(str(empty_file))
        assert isinstance(result, UtteranceList)
        assert len(result.utterances) == 0

    def test_missing_fields(self, tmp_path):
        """Test handling of JSON with missing fields"""
        # Create a JSON file with incomplete utterance data
        missing_fields_file = tmp_path / "missing_fields.json"
        missing_fields_file.write_text('''
        {
            "utterances": [
                {"id": "utt1"},
                {"ref_text": "Hello", "something_else": "value"}
            ]
        }
        ''')

        result = parse_utterance_list(str(missing_fields_file))
        assert isinstance(result, UtteranceList)
        assert len(result.utterances) == 0  # Should skip utterances with missing fields
