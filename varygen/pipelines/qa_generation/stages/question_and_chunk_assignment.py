# Question Generation Pipeline: QA and Chunk Assignment
# This module is responsible for filtering out unanswerable questions (outside
# of the scope of the document), generating new answers from the questions, and
# assigning the chunks from which they originate.

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

CACHE_FILENAME = "original_questions.csv"


class QuestionChunkAssignment(Stage):
    # pylint: disable=C0103
    """
    Handles Question-Chunk assignment based on the extracted facts/answers.
    """

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Sequence:
        1. Sequentially execute QA and check if the parsed questions, answers, and chunks have the same length as the original.
        2. Filter out answers with UNANSWERABLE, or chunks with [0].
        """
        LOG.info("Verify questions - chunk assignments.")

        if config.output_dir_from_previous_run:
            cache_from_previous_run = os.path.abspath(
                os.path.join("../..", config.output_dir_from_previous_run, CACHE_FILENAME)
            )
            if os.path.exists(cache_from_previous_run):
                LOG.info("Loading from cache '%s'.", cache_from_previous_run)
                final_questions_df = pd.read_csv(cache_from_previous_run)
                state.original_question_df = final_questions_df
                # Populate correctness evals with output of the generated questions
                state.correctness_evals_input_df = final_questions_df
                return
            LOG.warning("Provided cache path of '%s' cannot be found.", cache_from_previous_run)

        if state.injection_df is None:
            raise RuntimeError(
                "No domain injected questions is provided for the question chunk assignment verification."
            )
        if state.chunks_df is None:
            raise RuntimeError("No chunks is provided for the question chunk assignment verification.")

        # Prepare input
        df = state.injection_df
        df = df[
            [
                "filename",
                "final_questions",
                "answers",
            ]
        ]
        df = df.rename(columns={"final_questions": "questions"})

        dfs_dict = {key: group for key, group in df.groupby("filename", as_index=False)}  # pylint: disable=R1721
        # Prepare input
        evaluation_input = []
        chunks_df = state.chunks_df.copy()
        split_size = config.qa_generation.chunk_assignment_split_size
        for filename, df_grouped in dfs_dict.items():
            doc_chunks_df = chunks_df[chunks_df["filename"] == filename]
            evaluation_input.extend(
                [
                    (
                        filename,
                        math.floor(i / split_size) + 1,
                        df_grouped.iloc[i : i + split_size].reset_index(drop=True),
                        doc_chunks_df,
                        config.qa_generation,
                    )
                    for i in range(0, len(df_grouped), split_size)
                ]
            )
        evaluation_output: List[pd.DataFrame] = []
        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                evaluation_output.extend(executor.map(self._evaluate_question_chunk_assignment, evaluation_input))
        else:
            evaluation_output.extend([self._evaluate_question_chunk_assignment(input) for input in evaluation_input])

        if len(evaluation_output) > 0:
            final_questions_df = pd.concat(evaluation_output, ignore_index=True)
            final_questions_df.to_csv(CACHE_FILENAME, index=False)
            LOG.info("Final questions generated are saved at '%s'", CACHE_FILENAME)

            state.original_question_df = final_questions_df
            # Populate correctness evals with output of the generated questions
            state.correctness_evals_input_df = final_questions_df
        else:
            LOG.warning("No questions generated to be saved.")

    def _evaluate_question_chunk_assignment(self, args):
        filename, idx, local_df, doc_chunks_df, config = args

        # 1. Conduct QA and assign chunks
        LOG.info("Performing QA evaluation for %s (%s)", filename, idx)

        question_list = local_df["questions"].tolist()
        enumerated_question_list = "\n".join(f"{i+1}. {string}" for i, string in enumerate(question_list))
        messages = [
            {"role": "user", "content": self._qa_template(enumerated_question_list, doc_chunks_df["chunk"].tolist())}
        ]
        while True:
            LOG.debug("Sending %s (%s) question list and chunks to GPT for QA...", filename, idx)
            LOG.debug("Awaiting response...")
            response = self._evaluation_completion(messages, config)
            # Parses questions from text. Looks for lines in between 'Question: ' and '?'.
            # Returns a list of questions in order of appearance.
            questions = re.findall(r"Question: (.*?\?)", response.content)
            # Parses answers from text. Looks for lines starting with 'Answer: ' and ends with '\nEnd answer.'
            # Returns a list of answers in order of appearance.
            answers = re.findall(r"Answer: (.*?)\nEnd answer.", response.content, re.DOTALL)
            chunk_numbers = self._extract_chunk_numbers(response.content)

            LOG.debug("Input question list length for %s (%s): %s", filename, idx, len(question_list))
            LOG.debug("Number of extracted questions for %s (%s): %s", filename, idx, len(questions))
            LOG.debug("Number of extracted answers for %s (%s): %s", filename, idx, len(answers))
            LOG.debug("Number of extracted chunks for %s (%s): %s", filename, idx, len(chunk_numbers))

            if len(questions) == len(answers) == len(chunk_numbers) == len(question_list):
                local_df["new_questions"] = questions
                local_df["new_answers"] = answers
                local_df["chunk_numbers"] = chunk_numbers
                break
            LOG.error(
                "QA list length is not the same as input question list for %s (%s).\nOr the QA results were not parsed properly.\nRetrying...",
                filename,
                idx,
            )

        # 2. Filter the unanswerable questions
        filtered_df = local_df[local_df["new_answers"].str.lower() != "unanswerable"]
        # Convert [0] to string '0' and remove rows where 'chunk_numbers' is ['0']
        filtered_df = filtered_df[~filtered_df["chunk_numbers"].astype(str).isin(["0"])]
        filtered_df["chunk_numbers"] = filtered_df["chunk_numbers"].apply(lambda x: [i - 1 for i in x] if x else None)
        filtered_df.reset_index(drop=True, inplace=True)

        # 3. Format output to match variations expected input format
        # Explan the chunk numbers so that we can map them to the actual chunks
        filtered_df = filtered_df.explode("chunk_numbers")
        # Merge DataFrames on the exploded 'chunk_numbers' column
        merged_df = pd.merge(filtered_df, doc_chunks_df, left_on="chunk_numbers", right_on="index", how="left")

        # Group by the original DataFrame's index and aggregate 'chunk' and 'chunk_numbers' into lists
        grouped_df = merged_df.groupby(level=0).agg({"chunk": list, "chunk_numbers": list}).reset_index()

        # Drop unnecessary columns
        grouped_df.drop(columns=["index"], inplace=True)
        # Merge the aggregated values back to the original DataFrame
        result_df = pd.merge(filtered_df, grouped_df, left_index=True, right_index=True, how="left")

        result_df = result_df[["new_questions", "new_answers", "chunk_numbers_y", "chunk", "filename"]]
        result_df = result_df.rename(
            columns={
                "new_questions": "question",
                "new_answers": "answer",
                "chunk_numbers_y": "chunk_numbers",
                "chunk": "chunks",
            }
        )

        return result_df

    def _qa_template(self, question_list: str, chunks: List[str]) -> str:
        """
        Generate a textual prompt for guiding a question-answering task based on
        provided chunks of text.

        This function creates a structured prompt that includes the text chunks to
        be referenced, instructions for answering the questions, and a formatted
        list of questions to be answered.

        Parameters:
        - question_list (str): A multiline string where each line is a separate
        question to be answered.
        - chunks (List[str]): A list of string, where each string is a chunk of
        text that may contain information relevant to answering the questions.

        Returns:
        - str: A structured prompt as a single string that includes provided text
        chunks, the question list, and detailed answering guidelines.
        """

        prompt_header = """Read the following chunks for question-answering from a student later on.

"""

        question_list_section = f"""
Answer the following questions:
{question_list}
"""

        prompt_footer = """

Guidelines:
- You are a mentor answering a list of questions from a student.
- Reiterate the question before answering.
- If the chunks contain no information about the question in the answer field, write UNANSWERABLE.
- If the question is UNANSWERABLE, write down NONE in the source chunk field.
- In your answers, DO NOT INCLUDE NEW INFORMATION from any other document or your training other than information provided within the chunks.
- Answer all questions.

Use the following format for each question and answer pair in your response:

Question number: <insert question number>
Question: <copy the question from the given question list>
Answer: <answer the question using the provided information> \nEnd answer.
Source chunk of answer: <specify which chunk/s the answer comes from>
"""

        # Combine chunks and questoin list into one full prompt.
        fact_extraction_prompt = prompt_header
        for index, chunk in enumerate(chunks):
            opening_delimiter = f"\n***Chunk {index+1}***\n"
            closing_delimiter = f"\n***Chunk {index+1} End***\n"
            fact_extraction_prompt += opening_delimiter + chunk + closing_delimiter
        fact_extraction_prompt += question_list_section
        fact_extraction_prompt += prompt_footer
        return fact_extraction_prompt

    def _evaluation_completion(self, messages: List[Dict], config: QaGenerationConfig):
        # pylint: disable=R0801
        """A chat completion API call for QA."""

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=config.chunk_assignment_model_name,
                    temperature=float(config.chunk_assignment_temperature),
                    max_tokens=int(config.chunk_assignment_max_token),
                    top_p=float(config.chunk_assignment_top_p),
                    frequency_penalty=float(config.chunk_assignment_frequency_penalty),
                    presence_penalty=float(config.chunk_assignment_presence_penalty),
                    timeout=int(config.chunk_assignment_timeout),
                    messages=messages,
                )
                break
            except Exception as e:  # pylint: disable=W0718
                LOG.error("An error occurred:\n%s\nRetrying...", e)
        return response.choices[0].message

    def _extract_chunk_numbers(self, text: str) -> List[List[int]]:
        """Extracts numbers from lines starting with 'Source chunk of answer: '.
        Each set of numbers in a line become a sublist in the result.
        If 'NONE' is found, returns a list containing zero."""

        chunks = re.findall(r"Source chunk of answer: (.+)", text)
        numbers = []
        for chunk in chunks:
            # Check if the chunk contains 'NONE'
            if "NONE" in chunk:
                numbers.append([0])
            else:
                # Find all numbers in the chunk line and convert them to integers
                chunk_numbers = [int(num) for num in re.findall(r"Chunk (\d+)", chunk)]
                if chunk_numbers:
                    numbers.append(chunk_numbers)
        return numbers
