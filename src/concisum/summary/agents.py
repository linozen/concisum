import logging
from typing import List

from pydantic_ai import Agent

from concisum.summary.models import Utterance, UtteranceList, ChunkSummary, FullSummary
from concisum.config import model

LOG = logging.getLogger(__name__)

# Agent for summarizing individual chunks of transcript text
chunk_summarizer = Agent(
    model,
    output_type=ChunkSummary,
    system_prompt=(
        "Du bist ein Experte für die Zusammenfassung von psychotherapeutischen Sitzungen. "
        "Deine Aufgabe ist es, Abschnitte eines Therapietranskripts zwischen "
        "exakt EINER Therapeut*in und exakt EINER Klient*in zusammenzufassen. "
        "Konzentriere dich auf die wichtigsten Inhalte, Themen und Interventionen. "
        "Behalte wichtige Gefühle, Gedanken und Verhaltensweisen bei. "
        "Verfasse die Zusammenfassung auf Deutsch in der dritten Person und in einem professionellen Ton. "
        "Die Zusammenfassung sollte prägnant aber informativ sein."
    ),
)

# Agent for creating a comprehensive summary from individual chunk summaries
full_summarizer = Agent(
    model,
    output_type=FullSummary,
    system_prompt=(
        "Du bist ein Experte für die Zusammenfassung von psychotherapeutischen Sitzungen. "
        "Deine Aufgabe ist es, mehrere Teilzusammenfassungen zu einer kohärenten Gesamtzusammenfassung "
        "zu kombinieren. Erstelle eine strukturierte, zusammenhängende Zusammenfassung der gesamten Therapiesitzung. "
        "Die Zusammenfassung sollte einen klaren Überblick über die wichtigsten Themen, therapeutischen Interventionen "
        "und den Verlauf der Sitzung geben. Identifiziere Muster und wichtige Momente. "
        "Verfasse die Zusammenfassung in der dritten Person und in einem professionellen Ton. "
        "Die Gesamtzusammenfassung MUSS weniger als 250 Wörter enthalten."
    ),
)


class SummaryOrchestrator:
    """
    Orchestrates the hierarchical summarization process for therapy transcripts.
    """

    def __init__(
        self,
        chunk_size: int = 20,
        therapist_speaker_number: int = 0,
        generate_diagnosis: bool = False,
        use_rag: bool = True,
    ):
        """
        Initialize the summarizer with configurable chunk size.

        Args:
            chunk_size: Number of utterances per chunk
            therapist_speaker_number: Number of the speaker who is the therapist
            generate_diagnosis: Whether to generate a diagnosis alongside the summary
            use_rag: Whether to use the vector database (RAG) for diagnosis generation
        """
        self.chunk_size = chunk_size
        self.therapist_speaker_number = therapist_speaker_number
        self.generate_diagnosis = generate_diagnosis
        self.use_rag = use_rag

        # Import here to avoid circular imports
        if self.generate_diagnosis:
            from concisum.diagnosis.agents import DiagnosisOrchestrator

            self.diagnosis_orchestrator = DiagnosisOrchestrator(use_rag=self.use_rag)

    def _create_chunks(self, utterances: List[Utterance]) -> List[List[Utterance]]:
        """
        Split utterances into manageable chunks.

        Args:
            utterances: List of utterances from the transcript

        Returns:
            List of utterance chunks
        """
        chunks = [
            utterances[i : i + self.chunk_size]
            for i in range(0, len(utterances), self.chunk_size)
        ]

        # Log the total number of utterances and number of chunks created
        LOG.info(f"Total utterances: {len(utterances)}, Chunks created: {len(chunks)}")

        return chunks

    def _format_chunk_for_prompt(self, chunk: List[Utterance]) -> str:
        """
        Format a chunk of utterances into a readable prompt for the agent.

        Args:
            chunk: List of utterances in the chunk

        Returns:
            Formatted text representation of the chunk
        """
        formatted_text = ""
        for utt in chunk:
            formatted_text += f"{'[Therapeut*in]' if utt.speaker == self.therapist_speaker_number else '[Klient*in]'}: {utt.text}\n"
        return formatted_text

    async def summarize_chunk(self, chunk: List[Utterance]) -> ChunkSummary:
        """
        Summarize a single chunk of the transcript.

        Args:
            chunk: List of utterances in the chunk

        Returns:
            Summary of the chunk
        """
        formatted_chunk = self._format_chunk_for_prompt(chunk)
        prompt = (
            "Fasse den folgenden Teil eines therapeutischen Gesprächs auf Deutsch zusammen:\n\n"
            f"{formatted_chunk}"
        )

        result = await chunk_summarizer.run(prompt)
        return result.output

    async def summarize_full_transcript(
        self, chunk_summaries: List[ChunkSummary]
    ) -> FullSummary:
        """
        Create a comprehensive summary from individual chunk summaries.

        Args:
            chunk_summaries: List of summaries for each chunk

        Returns:
            Comprehensive summary of the entire transcript
        """
        combined_summaries = "\n\n".join(
            [
                f"Teil {i+1}:\n{summary.content}"
                for i, summary in enumerate(chunk_summaries)
            ]
        )

        prompt = (
            "Erstelle eine zusammenhängende deutsche Gesamtzusammenfassung aus den folgenden "
            "Teilzusammenfassungen einer psychotherapeutischen Sitzung. Achte darauf, dass du "
            "konsequent genderst. Schreibe also IMMER 'die Klient*in' und 'die Therapeut*in'. "
            "Es gibt max. 1 Therapeut*in und max. 1 Klient*in."
            "Die Zusammenfassung MUSS weniger als 300 Wörter enthalten:\n\n"
            f"{combined_summaries}"
        )

        print(prompt)
        result = await full_summarizer.run(prompt)

        # Verify word count and try again if necessary
        words = result.output.content.split()
        if len(words) > 300:
            LOG.warning(
                f"Summary too long ({len(words)} words). Requesting shorter version."
            )
            prompt = (
                "Erstelle eine kürzere Zusammenfassung der psychotherapeutischen Sitzung zwischen "
                "EINER Klient*in und EINER Therapeut*in. Nutze immer"
                "gendersensible Sprache und gehe immer nur von diesen beiden"
                "Gesprächspartner:innen aus."
                "Die Zusammenfassung darm MAXIMAL 300 Wörter enthalten. Die aktuelle Zusammenfassung hat {len(words)} Wörter "
                "und ist zu lang:\n\n"
                f"{result.output.content}"
            )
            result = await full_summarizer.run(prompt)

        return result.output

    async def process_transcript(self, utterance_list: UtteranceList) -> FullSummary:
        """
        Process a complete transcript through the hierarchical summarization pipeline.

        Args:
            utterance_list: List of utterances from the transcript

        Returns:
            Comprehensive summary of the entire transcript
        """
        # Split transcript into chunks
        chunks = self._create_chunks(utterance_list.utterances)

        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = await self.summarize_chunk(chunk)
            chunk_summaries.append(summary)

        # Create comprehensive summary from chunk summaries
        full_summary = await self.summarize_full_transcript(chunk_summaries)

        # Generate diagnosis if requested
        if self.generate_diagnosis:
            LOG.info("Generating diagnosis from transcript...")
            diagnosis_results = await self.diagnosis_orchestrator.process_transcript(
                chunks
            )
            full_summary.diagnosis = diagnosis_results.get("diagnosis")
            full_summary.symptoms = diagnosis_results.get("symptoms")
            LOG.info("Diagnosis generation complete")

        return full_summary
