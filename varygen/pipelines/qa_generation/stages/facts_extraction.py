# Query and Answer Synthesis: Extraction
# This module is responsible for reading all the chunks from a given document
# then extracts key facts from it.
# Then it checks the extracted facts and attempts to filter out duplicates.

import os
import re
import logging
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import openai
import pandas as pd

from varygen.config import ExperimentPipelineConfig, QaGenerationConfig
from varygen.pipelines.qa_generation.state import QaGenerationState

from surround import Stage

LOG = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION"]

CACHE_FILENAME = "facts_extraction.csv"


class FactsExtraction(Stage):
    # pylint: disable=C0103
    """
    Handles extraction from chunks to facts.
    """

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Sequence:
        1. Extracts facts from the given chunks. Prepares messages by using the
        above fact extraction template and inserts chunks and the required
        fact count to extract. Initially, extraction may not result in meeting
        the fact count. Therefore, the chat to ChatGPT is iterated until that
        count is met.

        2. Revises the extracted facts to remove duplicate facts, leaving only
        a list of unique facts. Output from ChatGPT is simply a list of integers
        corresponding to the members of the unique fact list.
        """

        LOG.info("Extracting facts.")

        if config.output_dir_from_previous_run:
            cache_from_previous_run = os.path.abspath(
                os.path.join("../..", config.output_dir_from_previous_run, CACHE_FILENAME)
            )
            if os.path.exists(cache_from_previous_run):
                LOG.info("Loading from cache '%s'.", cache_from_previous_run)
                facts_df = pd.read_csv(cache_from_previous_run)
                state.facts_df = facts_df
                return
            LOG.warning("Provided cache path of '%s' cannot be found.", cache_from_previous_run)

        if state.chunks_df is None:
            raise RuntimeError("No chunks is provided for extraction.")
        df = state.chunks_df.copy()
        LOG.info("Loaded %s chunks ready for extraction.", len(df))

        # Filter rows where it has no keywords to insert
        df = df[df["keywords_to_insert"].notna() & (df["keywords_to_insert"] != "")]

        # Prepare input
        files_to_process = df["filename"].unique().tolist()
        extraction_input = [(filename, df, config.qa_generation) for filename in files_to_process]
        extraction_output: List[Tuple[str, str, str, List[str]]] = []
        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                extraction_output.extend(executor.map(self._generate_question, extraction_input))
        else:
            extraction_output.extend([self._generate_question(input) for input in extraction_input])

        if len(extraction_output) > 0:
            # Define column names
            columns = ["filename", "keywords_to_insert", "document_summary", "facts"]
            # Create DataFrame
            facts_df = pd.DataFrame(extraction_output, columns=columns)
            # Explode the 'facts' column
            facts_df = facts_df.explode("facts")
            # Reset the index
            facts_df.reset_index(drop=True, inplace=True)

            # Update state values
            state.raw_extraction_output = extraction_output
            state.facts_df = facts_df

    def dump_output(self, state: QaGenerationState, _):
        """
        Exporting stage final output.
        """
        if state.facts_df is not None:
            state.facts_df.to_csv(CACHE_FILENAME, index=False)
            LOG.info("Extracted facts are cached at '%s'", CACHE_FILENAME)

    def _generate_question(self, args):
        filename, df, config = args
        # Prepare local chunks
        filtered_df = df[df["filename"] == filename]

        # Exit early when no chunks is found
        if len(filtered_df) == 0:
            LOG.warning("No chunks is found for '%s'", filename)
            return filename, "", "", []

        # 1. Fact extraction from given chunks.
        # Automatically determine facts given facts per chunk.
        fact_count = self._calculate_num_facts(filtered_df, config.facts_override, config.facts_per_chunk)
        LOG.info("Extraction facts %s for %s", fact_count, filename)

        # Prepare message to GPT with fact extraction template.
        # Inserted chunks and required fact count in the template.
        messages = [
            {"role": "user", "content": self._fact_extraction_template(filtered_df["chunk"].tolist(), fact_count)}
        ]

        # Fact extraction
        all_facts, document_summary = self._extract_facts(filename, messages, fact_count, config)
        LOG.debug("Extraction from '%s' document finished.", filename)

        # 2. Fact revision to remove semantically similar ones.
        # Re-enumerate the facts.
        facts_enumerated = "\n".join(f"{i+1}. {string}" for i, string in enumerate(all_facts))

        # Prepare new messages.
        messages = [{"role": "user", "content": self._duplicate_reduction_template(facts_enumerated)}]

        # Send to GPT for duplicate removal.
        while True:
            LOG.debug("Sending '%s' fact list to GPT for duplicate removal. Awating response...", filename)
            response = self._duplicate_removal_completion(messages, config)
            new_fact_numbers = self._new_fact_list_parser(response.content)
            # Checks if the response is valid.
            # If the return is "all", just return the original all_facts variable.
            if new_fact_numbers == [0]:
                LOG.debug("All '%s' questions are unique, no questions filtered.", filename)
                return filename, df["keywords_to_insert"].iloc[0], document_summary, all_facts

            if new_fact_numbers:
                break
            LOG.error("Fact numbers were not generated or successfully parsed.\nRetrying...")

        LOG.debug("Finished removing duplicates from '%s' fact list.", filename)
        LOG.debug("New '%s' fact list, filtered, numbers only: %s", filename, new_fact_numbers)

        # Filter out duplicates.
        cleaned_facts = []
        for number in new_fact_numbers:
            cleaned_facts.append(all_facts[number - 1])  # Use index to filter.
        LOG.debug("Completed facts extraction for %s", filename)

        return filename, df["keywords_to_insert"].iloc[0], document_summary, cleaned_facts

    def _calculate_num_facts(self, chunks: pd.DataFrame, facts_override: Optional[int], facts_per_chunk: int) -> int:
        """
        Calculates number of facts or returns the override number.
        """
        if facts_override:
            return facts_override
        return len(chunks) * facts_per_chunk

    def _fact_extraction_template(self, chunks: List[str], num_of_facts: int) -> str:
        """
        Constructs a prompt for extracting a specific number of detailed facts from
        document chunks.

        Parameters:
        - chunks (list of str): A list where each element contains a portion of
        text from the document.
        - num_of_facts (int): The number of detailed facts to be extracted from the
        chunks.

        Returns:
        - str: A string that includes the prompt instructions, format guidelines,
        all the chunks with delimiters, and a footer that outlines the desired fact
        enumeration format.
        """
        prompt_header = f"""Instructions: Read the following chunks that were obtained from the same document. Then extract {num_of_facts} detailed facts you have read from the document.

Guidelines:
- The extracted facts should be HIGHLY DETAILED AND SPECIFIC.
- EACH FACT MUST BE DIFFERENT from all the previous facts.
- DO NOT UNDER ANY CIRCUMSTANCE MENTION THE WORD "chunk" IN YOUR RESPONSE.
- You may go back and obtain information from previous chunks and check if you missed any information available for extraction.

"""

        prompt_footer = """

Further instructions:
First, read the document very carefully and generate a summary of the document so that you can extract the most useful facts. Think about the purpose of this document and overarching themes. Delimit it with "Summary: <insert summary> End summary."

Then generate the facts list.

Enumerate your extracted facts list. After the number, you must format the extracted fact like so: "1. Fact: <insert fact> End fact."
"""

        # Combine chunks into one full prompt.
        fact_extraction_prompt = prompt_header
        for index, chunk in enumerate(chunks):
            opening_delimiter = f"\n***Begin Chunk {index+1}***\n"
            closing_delimiter = f"\n***End Chunk {index+1}***\n"
            fact_extraction_prompt += opening_delimiter + chunk + closing_delimiter
        fact_extraction_prompt += prompt_footer
        return fact_extraction_prompt

    def _extracting_completion(self, messages: List[Dict], config: QaGenerationConfig):
        # pylint: disable=R0801
        """A chat completion API call for extracting facts."""

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=config.extraction_model_name,
                    temperature=float(config.extraction_temperature),
                    max_tokens=int(config.extraction_max_token),
                    top_p=float(config.extraction_top_p),
                    frequency_penalty=float(config.extraction_frequency_penalty),
                    presence_penalty=float(config.extraction_presence_penalty),
                    timeout=int(config.extraction_timeout),
                    messages=messages,
                )
                break
            except Exception as e:  # pylint: disable=W0718
                LOG.error("An error occurred:\n%s\nRetrying...", e)
        return response.choices[0].message

    def _extract_facts(self, filename, messages, fact_count, config: QaGenerationConfig):
        """
        Fact extraction loop to get facts from document chunks.
        Also returns the document summary.
        """
        check_document_summary = True
        all_facts = []
        while True:
            LOG.debug("Sending '%s' chunks to GPT for fact extraction.", filename)
            LOG.debug("May take a minute depending on document length.")
            LOG.debug("Awaiting response...")
            response = self._extracting_completion(messages, config)

            # The function searches the text for patterns that start with 'Fact: ' and end with 'End fact.',
            # then extracts the text in between these markers.
            # The extracted facts are returned as a list ordered by their appearance in the text.
            all_facts.extend(re.findall(r"Fact: (.*?)End fact.", response.content))

            # Check summary is generated.
            if check_document_summary:
                # Parses summary from text. Looks for lines starting with 'Summary: ' and ends with '\nEnd summary.'
                # Returns the summary in string form.
                document_summary = re.findall(r"Summary: (.*?)End summary.", response.content)

                if not document_summary:
                    LOG.error("Summary not found. Resending initial prompt.")
                    continue

                document_summary = document_summary[0]
                LOG.debug("Summary found for '%s'.", filename)
                check_document_summary = False

            # Checking fact count is reached.
            if len(all_facts) >= fact_count:
                LOG.debug("Cumulative number of facts extracted for '%s': %s", filename, len(all_facts))
                LOG.debug("Fact count reached for '%s'.", filename)
                break

            LOG.debug("Cumulative number of facts extracted for '%s': %s", filename, len(all_facts))
            LOG.debug(
                "Fact count of %s not yet reached for '%s'.\nResending full chat and adding user instructions.",
                fact_count,
                filename,
            )
            # Append previous messages so GPT is less likely to repeat facts.
            # Insert user instruction to keep extracting more facts.
            messages.extend(
                [
                    {"role": response.role, "content": response.content},
                    {
                        "role": "user",
                        "content": f"Continue extracting until {fact_count} facts have been extracted. Use the same output format as per the instructions. Remember - DO NOT REPEAT already extracted facts.",
                    },
                ]
            )
        return all_facts, document_summary

    def _duplicate_reduction_template(self, fact_list: str) -> str:
        """Construct a prompt tailored for identifying and eliminating duplicate
        facts from a list.

        Parameters:
        fact_list (str): A string containing a numbered list of facts separated by
        new lines.

        Returns:
        str: A prompt string with instructions and the passed `fact_list` for the
        reviewer to process and indicate the unique facts, formatted for
        subsequent parsing.

        The returned prompt adds instructions on formatting the output as a Python
        list of fact numbers wrapped with the phrase "New fact list: " for easy
        identification and parsing.
        """
        prompt_template = f"""The following list of facts was obtained from one document. Review them and look for EXACTLY equivalent facts. If found, remove the duplicates such that ONE of the facts remain.

First give a short reasoning for your found duplicates, mentioning the question numbers. Remember to keep one instance of each semantically identical fact. In your reasoning, say which one you will keep and which ones you will remove.

At the end of your response, use the format "New fact list: <insert fact list>" for your Python list of numbers corresponding to the filtered unique facts. If no duplicates were found and removed, use "New fact list: ["ALL"]"

Begin list:
{fact_list}
"""
        return prompt_template

    def _duplicate_removal_completion(self, messages: List[Dict], config: QaGenerationConfig):
        # pylint: disable=R0801
        """A chat completion API call for removing duplicate facts."""
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=config.duplicate_removal_model_name,
                    temperature=float(config.duplicate_removal_temperature),
                    max_tokens=int(config.duplicate_removal_max_token),
                    top_p=float(config.duplicate_removal_top_p),
                    frequency_penalty=float(config.duplicate_removal_frequency_penalty),
                    presence_penalty=float(config.duplicate_removal_presence_penalty),
                    timeout=int(config.duplicate_removal_timeout),
                    messages=messages,
                )
                break
            except Exception as e:  # pylint: disable=W0718
                LOG.error("An error occurred:\n%s\nRetrying...", e)
        return response.choices[0].message

    def _new_fact_list_parser(self, text: str) -> List | None:
        """
        Parse a string to extract a list of unique numerical facts.

        The function searches for a specific pattern indicating a list of facts
        within the input text, extracts that list, and then processes it into a
        list of integers. It is assumed that the list of facts in the text is
        represented as 'New fact list: [fact1, fact2, fact3, ...]', where each fact
        is a unique integer.

        Parameters:
        - text (str): A string potentially containing a list of facts.

        Returns:
        - list: A list of integers representing the extracted unique facts. If no
        list is found in the input text, returns None.
        """

        # Check if the fact list needs no filtering
        match = re.search(r'New fact list: \["ALL"\]', text)
        if match:
            return [0]

        match = re.search(r"New fact list: \[(.*?)\]", text)
        if match:
            # Extract the matched group contents and split by comma and space.
            fact_list_str = match.group(1)
            # Convert to a list of integers.
            new_fact_list = [int(fact.strip()) for fact in fact_list_str.split(",")]
            return new_fact_list
        return None
