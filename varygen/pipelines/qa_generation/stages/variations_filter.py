import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import openai
import pandas as pd


from varygen.config import ExperimentPipelineConfig
from varygen.pipelines.qa_generation.state import QaGenerationState
from varygen.pipelines.qa_generation.common import load_cache, fix_response_when_required

from surround import Stage

LOG = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION"]

EXPECTED_OUTPUT_HEADERS = ["original_question", "new_question", "answer_x", "chunk_numbers", "chunks", "filename"]

PROMPT = """Question: `{original_question}`
Answer: `{answer}`
Question candidates:
```
{new_questions_string}
```


You will pick questions from the `Question candidates` that have the same meaning as the `Original question` and CAN be answered by the `Provided answer`.

Guidelines:
- Exclude all questions that don't fit in the selection criteria.

Answer in JSON. The JSON should be a list of just the filtered questions."""


class VariationsFilter(Stage):
    # pylint: disable=R0801,C0103
    """
    Filter unrelated variations to reduce the outcome of hallucinations from the LLM models.
    """

    def initialise(self, config: ExperimentPipelineConfig):
        # pylint: disable=W0201
        """
        Set up required variables.
        """
        if not config.variation_filter_generation.cache_dir_path:
            raise RuntimeError("Cache directory is not specified.")
        if not config.variation_filter_generation.intermediate_output_prefix:
            raise RuntimeError("Prefix is not specified.")

        self.cache_dir_path = config.variation_filter_generation.cache_dir_path
        self.intermediate_output_prefix = config.variation_filter_generation.intermediate_output_prefix
        self.cache_dir_from_previous_run = None
        if config.output_dir_from_previous_run:
            self.cache_dir_from_previous_run = os.path.abspath(
                os.path.join(
                    "../..", config.output_dir_from_previous_run, config.variation_filter_generation.cache_dir_path
                )
            )

        # Prepare OpenAI configuration
        self.model_name = config.variation_filter_generation.model_name
        self.temperature = config.variation_filter_generation.temperature

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Generate variations foe every single question-answer pair.
        """
        LOG.info("Initiate filtering stage")

        if state.original_question_df is None:
            raise RuntimeError("No original QA sets is provided for variation filter.")

        loaded_dfs, df = load_cache(
            self.cache_dir_path, self.cache_dir_from_previous_run, self.intermediate_output_prefix
        )

        LOG.info("Filtering question variations generated.")
        # Get filtered original questions
        filtered_questions = list(set(df["original_question"].values))
        LOG.info("Filtered %s question(s)", len(filtered_questions))

        # Prepare input from the original questions
        original_input = [
            (idx, question, answer)
            for idx, (question, answer, _, __, ___) in state.original_question_df.iterrows()
            if question not in filtered_questions
        ]
        if state.variation_list_output is None:
            raise RuntimeError("Required variation generation output is not available.")
        # Preparing add the generated question variations
        filter_input = [
            original_input[idx] + (output["new_question"].tolist(),)
            for idx, output in enumerate(state.variation_list_output)
            if idx in [idx for idx, _, _ in original_input]
        ]
        LOG.info("Missing %s index(s): %s", len(filter_input), ", ".join([str(item[0]) for item in filter_input]))

        LOG.info("Filtering variations from %s original question(s). This might take a while...", len(filter_input))
        # Use multiprocessing to filter the generated variation
        filter_output = loaded_dfs

        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                filter_output.extend(executor.map(self._filter_variations, filter_input))
        else:
            filter_output.extend([self._filter_variations(input) for input in filter_input])
        LOG.info("Finished filtering variations.")

        # # Concatenate the loaded DataFrames
        if filter_output:
            merged_dfs = pd.concat(filter_output, ignore_index=True)

            # Append the loaded DataFrames to the existing DataFrame
            df = pd.concat([df, merged_dfs], ignore_index=True)
            state.filter_df = df

            # Prepare the final DF for evaluation
            final_output_path = "new_questions.csv"
            final_questions_df = pd.merge(
                state.original_question_df,
                state.filter_df,
                left_on="question",
                right_on="original_question",
                how="inner",
            )
            final_questions_df = final_questions_df[EXPECTED_OUTPUT_HEADERS]
            final_questions_df = final_questions_df.rename(columns={"answer_x": "answer"})
            final_questions_df.to_csv(final_output_path, index=False)
            state.final_questions_df = final_questions_df
            # Populate correctness evals with output of the generated questions
            state.correctness_evals_input_df = final_questions_df

    def dump_output(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Exporting stage final output.
        """
        final_filename = f"{config.variation_filter_generation.intermediate_output_prefix}_final.csv"
        if state.filter_df is not None:
            state.filter_df.to_csv(final_filename, index=False)
            LOG.info("Filtered variations output is cached at '%s'", final_filename)

    def _filter(self, original_question: str, answer: str, new_questions: list) -> str:
        new_questions_string = "\n".join(new_questions)
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT.format(
                        original_question=original_question, answer=answer, new_questions_string=new_questions_string
                    ),
                }
            ],
        )
        return completion.choices[0].message.content

    # Define a function to process the code for a given question and answer
    def _filter_variations(self, args) -> pd.DataFrame:
        """
        Going through each original question to filter the variations created from it.
        """
        idx, original_question, answer, new_questions = args
        LOG.debug("Initiating %s) '%s' question's variations filtering.", idx, original_question)
        try:
            response = self._filter(original_question, answer, new_questions)
            new_questions = json.loads(response)
            temp_df = pd.DataFrame(list(set(new_questions["questions"])), columns=["new_question"])
        except Exception:  # pylint: disable=W0718
            try:
                new_questions = json.loads(fix_response_when_required(response))
                temp_df = pd.DataFrame(list(set(new_questions["questions"])), columns=["new_question"])
            except Exception as exception:  # pylint: disable=W0718
                LOG.error(exception)
        temp_df["original_question"] = original_question
        temp_df["answer"] = answer

        # Cache DF as intermediate results
        cache_path = f"{self.cache_dir_path}/{self.intermediate_output_prefix}_{idx}.csv"
        temp_df.to_csv(cache_path, index=False)
        LOG.debug("Cache %s) '%s' filtered question's variations at '%s'", idx, original_question, cache_path)
        return temp_df
