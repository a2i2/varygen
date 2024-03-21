# Query and Answer Synthesis: Generation
# This module is responsible for generating questions from a given answer list.

import os
import re
import logging
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import openai
import pandas as pd


from varygen.config import ExperimentPipelineConfig, QaGenerationConfig
from varygen.pipelines.qa_generation.state import QaGenerationState

from surround import Stage

LOG = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION"]

CACHE_FILENAME = "questions_generation.csv"


class QuestionsGeneration(Stage):
    # pylint: disable=C0103
    """
    Handles the questions generation from the extracted facts.
    """

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Sequence:
        1. Generate the questions from the groups. Check the number of parsed
        questions or answers before moving on to the next group of facts.
        """

        LOG.info("Generating questions from facts.")

        if config.output_dir_from_previous_run:
            cache_from_previous_run = os.path.abspath(
                os.path.join("../..", config.output_dir_from_previous_run, CACHE_FILENAME)
            )
            if os.path.exists(cache_from_previous_run):
                LOG.info("Loading from cache '%s'.", cache_from_previous_run)
                qa_df = pd.read_csv(cache_from_previous_run)
                state.qa_df = qa_df
                return
            LOG.warning("Provided cache path of '%s' cannot be found.", cache_from_previous_run)

        if state.facts_df is None:
            raise RuntimeError("No facts is provided for extraction.")

        # pylint: disable=R1721
        # Group by 'filename' and create a dictionary with 'filename' as keys and DataFrames as values
        dfs_dict = {key: group for key, group in state.facts_df.groupby("filename", as_index=False)}

        # Prepare input
        generation_input = []
        split_size = config.qa_generation.generation_split_size
        # Divides the fact list into sets of 10 to make the task easier for ChatGPT.
        # This aims to reduce the probability that ChatGPT will under deliver the amount to generate.
        for filename, df_grouped in dfs_dict.items():
            keywords = df_grouped["keywords_to_insert"].iloc[0]
            summary = df_grouped["document_summary"].iloc[0]
            facts = df_grouped["facts"].tolist()
            divided_facts = [facts[i : i + split_size] for i in range(0, len(facts), split_size)]
            generation_input.extend(
                [
                    (filename, keywords, summary, short_facts, idx + 1, config.qa_generation)
                    for idx, short_facts in enumerate(divided_facts)
                ]
            )

        generation_output: List[List[Tuple[str, str, str, str]]] = []
        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                generation_output.extend(executor.map(self._generate_questions, generation_input))
        else:
            generation_output.extend([self._generate_questions(input) for input in generation_input])

        if len(generation_output) > 0:
            # Flatten results to be parsed into a DF
            flatten_outputs: List[Tuple[str, str, str, str]] = []
            for questions in generation_output:
                flatten_outputs.extend(questions)
            # Define column names
            columns = ["filename", "keywords_to_insert", "questions", "answers"]
            # Create DataFrame
            qa_df = pd.DataFrame(flatten_outputs, columns=columns)

            # Update state values
            state.raw_generation_output = generation_output
            state.qa_df = qa_df

    def _generate_questions(self, args):
        filename, keywords, summary, short_facts, idx, config = args
        LOG.info("Performing generation for %s (%s).", filename, idx)

        enumerated_fact_list = "\n".join(f"{i+1}. {string}" for i, string in enumerate(short_facts))
        messages = [{"role": "user", "content": self._question_generation_template(enumerated_fact_list, summary)}]
        # Check if the correct number of questions and answers are extracted
        while True:
            LOG.debug("Sending %s (%s) fact list to GPT for question generation...", filename, idx)
            LOG.debug("Awaiting response...")
            response = self._generation_completion(messages, config)
            # Parses questions from text. Looks for lines in between 'Question: ' and '?'.
            # Returns a list of questions in order of appearance.
            questions = re.findall(r"Question: (.*?\?)", response.content)
            # Parses answers from text. Looks for lines in between 'Answer: ' and a new line character.
            # Returns a list of questions in order of appearance.
            answers = re.findall(r"Answer: (.*?)\n", response.content)

            # Check if the length of the questions and answers generated
            # are according to the input fact list
            LOG.debug("Questions generated for %s (%s): %s", filename, idx, len(questions))
            LOG.debug("Answers reiterated for %s (%s): %s", filename, idx, len(answers))
            LOG.debug("Fact list input length for %s (%s): %s", filename, idx, len(short_facts))
            if len(questions) == len(answers) == len(short_facts):
                break

            LOG.error(
                "Number of generated questions does not match the fact list for %s (%s).\nOr the output is not parsed well.\nRetrying...",
                filename,
                idx,
            )

        return [(filename, keywords, question, answers[i]) for i, question in enumerate(questions)]

    def dump_output(self, state: QaGenerationState, _):
        """
        Exporting stage final output.
        """
        if state.qa_df is not None:
            state.qa_df.to_csv(CACHE_FILENAME, index=False)
            LOG.info("Extracted facts are cached at '%s'", CACHE_FILENAME)

    def _question_generation_template(self, fact_list: List[str], context: str) -> str:
        """
        Constructs a series of questions corresponding to a provided list of facts.

        Parameters:
        - fact_list (List[str]): A list of strings, each of which is a fact that
        the generated questions should target.
        - context (str): A string containing the summary of the document obtained
        from `extraction.py`.

        Returns:
        - str: A string containing the prompt to generate questions from facts.
        """
        prompt_template = f"""Create questions whose answers are exactly the answers in the following list:

{fact_list}

The context of the document from which these facts are extracted is given here.
Context: {context}

Guidelines:
- Make questions for all answers in the given list.
- First reiterate the answer, then create the question.
- The questions you generate should not require information outside of the given fact to answer it.
- The questions must be very specific and detailed about what they are asking. No broad questions.

Use the following format for each answer-question pair you generate, I will need them for parsing your output:
Answer: <reiterate the answer>
Question: <create the question>
"""
        return prompt_template

    def _generation_completion(self, messages: List[Dict], config: QaGenerationConfig):
        # pylint: disable=R0801
        """A chat completion API call for generating questions from facts."""

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=config.generation_removal_model_name,
                    temperature=float(config.generation_removal_temperature),
                    max_tokens=int(config.generation_removal_max_token),
                    top_p=float(config.generation_removal_top_p),
                    frequency_penalty=float(config.generation_removal_frequency_penalty),
                    presence_penalty=float(config.generation_removal_presence_penalty),
                    timeout=int(config.generation_removal_timeout),
                    messages=messages,
                )
                break
            except Exception as e:  # pylint: disable=W0718
                LOG.error("An error occurred:\n%s\nRetrying...", e)
        return response.choices[0].message
