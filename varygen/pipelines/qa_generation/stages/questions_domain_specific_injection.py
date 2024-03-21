# Query and Answer Synthesis: Domain Specific Injection
# This module is responsible for inserting the filename into the question list.
# Then changes the format of the questions so that questions are from the
# perspective of the student.

import os
import re
import math
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import openai
import pandas as pd


from varygen.config import ExperimentPipelineConfig, QaGenerationConfig
from varygen.pipelines.qa_generation.state import QaGenerationState

from surround import Stage

LOG = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION"]

CACHE_FILENAME = "domain_injected_questions.csv"


class DomainSpecificInjection(Stage):
    # pylint: disable=C0103
    """
    Handles domain specific injection to the generated questions.
    """

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """Sequence:
        1. Change the perspective into first-person.
        Check the output modified list is the same as original question list.
        2. Insert the filename into the questions.
        Check the output modified list is the same as original question list.
        """
        LOG.info("Injecting domain specific information.")

        if config.output_dir_from_previous_run:
            cache_from_previous_run = os.path.abspath(
                os.path.join("../..", config.output_dir_from_previous_run, CACHE_FILENAME)
            )
            if os.path.exists(cache_from_previous_run):
                LOG.info("Loading from cache '%s'.", cache_from_previous_run)
                injection_df = pd.read_csv(cache_from_previous_run)
                state.injection_df = injection_df
                return
            LOG.warning("Provided cache path of '%s' cannot be found.", cache_from_previous_run)

        if state.qa_df is None:
            raise RuntimeError("No QA sets is provided for the domain injection.")

        # pylint: disable=R1721
        dfs_dict = {key: group for key, group in state.qa_df.groupby("filename", as_index=False)}
        # Prepare input
        injection_input = []
        split_size = config.qa_generation.injection_split_size
        for filename, df_grouped in dfs_dict.items():
            injection_input.extend(
                [
                    (
                        filename,
                        math.floor(i / split_size) + 1,
                        df_grouped.iloc[i : i + split_size].reset_index(drop=True),
                        config.qa_generation,
                    )
                    for i in range(0, len(df_grouped), split_size)
                ]
            )

        injection_output: List[pd.DataFrame] = []
        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                injection_output.extend(executor.map(self._domain_injection, injection_input))
        else:
            injection_output.extend([self._domain_injection(input) for input in injection_input])

        if len(injection_output) > 0:
            injection_df = pd.concat(injection_output, ignore_index=True)
            state.injection_df = injection_df

    def dump_output(self, state: QaGenerationState, _):
        """
        Exporting stage final output.
        """
        if state.injection_df is not None:
            state.injection_df.to_csv(CACHE_FILENAME, index=False)
            LOG.info("Domain injected questions are cached at '%s'", CACHE_FILENAME)

    def _domain_injection(self, args):
        filename, idx, local_df, config = args
        # 1. Change the perspective into first-person.
        LOG.info("Performing first-person modification for %s (%s).", filename, idx)

        # Inserting question list with number <= 10 into messages.
        original_question_list = "\n".join(
            f"{i+1}. {string}" for i, string in enumerate(local_df["questions"].tolist())
        )
        messages = [{"role": "user", "content": self._firstperson_perspective_template(original_question_list)}]

        # Loop to ensure correct number of questions is parsed.
        while True:
            LOG.debug("Sending question list to GPT for %s (%s) first-person modification...", filename, idx)
            LOG.debug("Awaiting response...")
            response = self._injection_completion(messages, config)
            # Parses numbered questions from text. Looks for text that starts with a
            # number followed by a dot and a whitespace, and ends with a '?'.
            # Returns a list of questions including question marks in order of appearance.
            first_person_questions = re.findall(r"\d+\.\s(.*?\?)", response.content)

            LOG.debug("First person question list length for %s (%s): %s", filename, idx, len(first_person_questions))
            if len(first_person_questions) == len(local_df):
                local_df["first_person_questions"] = first_person_questions
                break
            LOG.error("Modified question list is not the same as original for %s (%s).\nRetrying...", filename, idx)

        # 2. Insert the keywords into the question using the prompt template.
        LOG.info("Performing keywords insertion for %s (%s).", filename, idx)

        # Inserting question list with number <= 10 into messages.
        keywords = local_df["keywords_to_insert"].iloc[0]
        first_person_question_list = "\n".join(f"{i+1}. {string}" for i, string in enumerate(first_person_questions))
        messages = [
            {"role": "user", "content": self._filename_insertion_template(first_person_question_list, keywords)}
        ]

        # Loop to ensure correct number of questions is parsed.
        while True:
            LOG.debug("Sending question list to GPT for %s (%s) filename insertion...", filename, idx)
            LOG.debug("Awaiting response...")
            response = self._injection_completion(messages, config)
            # Parses numbered questions from text. Looks for text that starts with a
            # number followed by a dot and a whitespace, and ends with a '?'.
            # Returns a list of questions including question marks in order of appearance.
            final_questions = re.findall(r"\d+\.\s(.*?\?)", response.content)

            LOG.debug("Domain injected question list length for %s (%s): %s", filename, idx, len(final_questions))
            if len(final_questions) == len(local_df):
                local_df["final_questions"] = final_questions
                break
            LOG.error("Modified question list is not the same as original for %s (%s).\nRetrying...", filename, idx)
        return local_df

    def _firstperson_perspective_template(self, question_list: List[str]) -> str:
        """
        Constructs a prompt in order to change the list of questions to be in a
        first-person perspective.

        Parameters:
        - question_list (List[str]): A list of strings, each of which is a question
        that may be modified to an appropriate first-person perspective.

        Returns:
        - str: A string containing the prompt to modify the question list to
        student perspective.
        """
        prompt_template = f"""The following list of questions are questions students have of a particular course document.
WHERE APPROPRIATE, change the phrasing of the question to more fit this context.
For example, if the question's subject is the "student", change it to a first-person perspective.
You may only need to change some, or even none of the questions.
The answer to the questions should be the same; DO NOT CHANGE the meaning of the question.
You only need to output a list of the modified questions and enumerate them.

{question_list}
"""
        return prompt_template

    def _injection_completion(self, messages: List[Dict], config: QaGenerationConfig):
        # pylint: disable=R0801
        """A chat completion API call for modifying questions."""

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=config.injection_model_name,
                    temperature=float(config.injection_temperature),
                    max_tokens=int(config.injection_max_token),
                    top_p=float(config.injection_top_p),
                    frequency_penalty=float(config.injection_frequency_penalty),
                    presence_penalty=float(config.injection_presence_penalty),
                    timeout=int(config.injection_timeout),
                    messages=messages,
                )
                break
            except Exception as e:  # pylint: disable=W0718
                LOG.error("An error occurred:\n%s\nRetrying...", e)
        return response.choices[0].message

    def _filename_insertion_template(self, question_list: str, filename: str) -> str:
        """
        Constructs a prompt for the insertion of the filename into the question.
        """
        prompt_template = f"""The following list of student questions are for obtaining information from a document called "{filename}".
Append the questions, without changing the meaning or any key details, so that they all refer to "{filename}".
You only need to output a list of the modified questions and enumerate them.

Question list:
{question_list}
"""
        return prompt_template
