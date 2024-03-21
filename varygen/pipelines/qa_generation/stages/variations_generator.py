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

# General prompt
PROMPT = """Question: `{question}`
Answer: `{answer}`

You will generate variations ONLY from the provided Question above and ONLY use the Answer to constraint the generated questions.

Variation could be any of these, but not limited to:
- Rephrasing with synonyms: Changing one or more words with a different word(s) that have similar meaning.
- Changing the sentence structure.
- Using passive voice.
- Using different Question words: What, When, How, Is, Do/Does, Can/Could, Will/Would.
- Converting it to indirect question.
- Generalising the question.
- Simplifying the question: A shorter more direct question.
- Adding context or rationale: Give the short reason why the question is asked.
- Reframing with a different verb or action.
- Using slang or informal Language.

Guidelines:
- Do NOT use keywords from the Answer when generating new questions!
- The generated questions can be rephrased again just with synonyms as new questions.
- No exact same generated question.
- Use different Question words to rephrase the questions.
- The new questions can be answered with the same Answer.
- The new questions need to have the same meaning as the original Question.
- ONLY create variations from the words in the original question!

Answer in JSON. The JSON should be a list (up to 200) of just the new questions."""


class VariationsGenerator(Stage):
    # pylint: disable=R0801,C0103
    """
    Generate question variations by using OpenAI.
    Given the current case study, we explored the use of 2 different prompts (generic vs assignment specific domain).
    """

    def initialise(self, config: ExperimentPipelineConfig):
        # pylint: disable=W0201
        """
        Set up required variables.
        """
        if not config.variation_generation.cache_dir_path:
            raise RuntimeError("Cache directory is not specified.")
        if not config.variation_generation.intermediate_output_prefix:
            raise RuntimeError("Prefix is not specified.")

        self.cache_dir_path = config.variation_generation.cache_dir_path
        self.intermediate_output_prefix = config.variation_generation.intermediate_output_prefix
        self.cache_dir_from_previous_run = None
        if config.output_dir_from_previous_run:
            self.cache_dir_from_previous_run = os.path.abspath(
                os.path.join("../..", config.output_dir_from_previous_run, config.variation_generation.cache_dir_path)
            )

        # Prepare OpenAI configuration
        self.model_name = config.variation_generation.model_name
        self.temperature = config.variation_generation.temperature
        self.frequency_penalty = config.variation_generation.frequency_penalty

    def operate(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Generate variations for every single question-answer pair.
        """
        if state.original_question_df is None:
            raise RuntimeError("Original question dataframe is required.")

        original_df = state.original_question_df.copy()
        LOG.info("Generate variations for %s questions.", len(original_df))

        loaded_dfs, df = load_cache(
            self.cache_dir_path, self.cache_dir_from_previous_run, self.intermediate_output_prefix
        )

        LOG.info("Starting questions variation generation.")
        # Get asked original questions
        asked_questions = list(set(df["original_question"].values))
        LOG.info("Asked %s question(s).", len(asked_questions))

        # Create a list of tuples with (idx, question, answer) for rows where the question is not in asked_questions
        variation_input = [
            (idx, question, answer)
            for idx, (question, answer, _, __, ___) in original_df.iterrows()
            if question not in asked_questions
        ]
        LOG.info(
            "Missing %s index(s): %s",
            len(variation_input),
            ", ".join([str(item[0]) for item in variation_input]),
        )

        LOG.info("Generating variations for %s question(s). This might take a while...", len(variation_input))
        # Include pre-loaded DFs
        variation_output = loaded_dfs

        if config.parallel:
            with ThreadPoolExecutor(max_workers=config.number_of_workers) as executor:
                # Use multiprocessing to process generate the variation
                variation_output.extend(executor.map(self._create_variations, variation_input))
        else:
            variation_output.extend([self._create_variations(input) for input in variation_input])
        LOG.info("Finished generating variations.")

        # Concatenate the loaded DataFrames
        if variation_output:
            merged_dfs = pd.concat(variation_output, ignore_index=True)

            # Append the loaded DataFrames to the existing DataFrame
            df = pd.concat([df, merged_dfs], ignore_index=True)
            state.variation_df = df
            state.variation_list_output = variation_output
        else:
            LOG.warning("No variations generated to be saved.")

    def dump_output(self, state: QaGenerationState, config: ExperimentPipelineConfig):
        """
        Exporting stage final output.
        """
        final_filename = f"{config.variation_generation.intermediate_output_prefix}_final.csv"
        if state.variation_df is not None:
            state.variation_df.to_csv(final_filename, index=False)
            LOG.info("Variation output is cached at '%s'", final_filename)

    def _generate(self, question, answer) -> str:
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT.format(question=question, answer=answer),
                }
            ],
        )
        return completion.choices[0].message.content

    def _create_variations(self, args) -> pd.DataFrame:
        """
        Going through each original question - answer pairs to generate variation of the questions.
        """
        idx, question, answer = args
        LOG.debug("Initiating %s) '%s' question's variations generation.", idx, question)
        try:
            response = self._generate(question, answer)
            new_questions = json.loads(response)
            temp_df = pd.DataFrame(list(set(new_questions["questions"])), columns=["new_question"])
        except Exception:  # pylint: disable=W0718
            try:
                new_questions = json.loads(fix_response_when_required(response))
                temp_df = pd.DataFrame(list(set(new_questions["questions"])), columns=["new_question"])
            except Exception as exception:  # pylint: disable=W0718
                LOG.error(exception)
        temp_df["original_question"] = question
        temp_df["answer"] = answer

        # Cache DF as intermediate results
        cache_path = f"{self.cache_dir_path}/{self.intermediate_output_prefix}_{idx}.csv"
        temp_df.to_csv(cache_path, index=False)
        LOG.debug("Cache %s) '%s' question's variations at '%s'", idx, question, cache_path)
        return temp_df
