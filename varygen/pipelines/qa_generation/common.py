import os
import re
import json
import shutil
from typing import Optional
import pandas as pd

# pylint: disable=C0103


def extract_index(filename):
    """
    Extract index from filename to load the files from a different stage.
    """
    return int(filename.split("_")[-1].split(".")[0])


def load_cache(cache_dir_path: str, cache_dir_from_previous_run: Optional[str], file_prefix: str) -> pd.DataFrame:
    """
    Load cache from previous runs to skip stages
    """
    # Prepare new cache directory
    os.makedirs(cache_dir_path, exist_ok=True)

    # Chec existing cache
    if cache_dir_from_previous_run and os.path.exists(cache_dir_from_previous_run):
        old_cache_files = os.listdir(cache_dir_from_previous_run)
        # Copy them to the new cache directory
        for file in old_cache_files:
            source_path = os.path.join(cache_dir_from_previous_run, file)
            destination_path = os.path.join(cache_dir_path, file)
            shutil.copy(source_path, destination_path)

    file_list = os.listdir(cache_dir_path)
    if len(file_list) == 0:
        # Exit early when no files in the cahce dir
        return [], pd.DataFrame(columns=["original_question", "new_question", "answer"])

    # Filter for CSV files starting with variation prefix
    csv_files = sorted([f for f in file_list if f.startswith(file_prefix)], key=extract_index)

    # Initialize an empty list to store loaded DataFrames
    loaded_dfs = []
    df = pd.DataFrame(columns=["original_question", "new_question", "answer"])

    # Load the CSV files into DataFrames
    for csv_file in csv_files:
        temp_df = pd.read_csv(os.path.join(cache_dir_path, csv_file))
        # Reorder the columns in the temp_df DataFrame
        temp_df = temp_df[["original_question", "new_question", "answer"]]
        loaded_dfs.append(temp_df)

    if loaded_dfs:
        merged_dfs = pd.concat(loaded_dfs, ignore_index=True)

        # Append the loaded DataFrames to the existing DataFrame
        df = pd.concat([df, merged_dfs], ignore_index=True)

    return loaded_dfs, df


def fix_response_when_required(response):  # pylint: disable=R0911
    """
    Parser to convert OpenAI responses with a few possible scenarios that have been encountered so far.
    """
    # Handle incomplete response

    response = response.strip()
    pattern = r'"question": "([^"]+)"'

    # Use re.findall to extract the questions into a list
    questions = re.findall(pattern, response)
    if len(questions) > 0:
        return json.dumps({"questions": questions})

    response_length = len(response)
    closing_brackets_index = response.rfind("]}")
    if closing_brackets_index > 0:
        return response  # Response should be full

    starting_curly_bracket_index = response.find("{")
    if starting_curly_bracket_index < 0:
        # response is just a list
        closing_bracket_index = response.rfind("]")
        if closing_bracket_index > 0:
            return json.dumps({"questions": json.loads(response)})  # Wrap full response in the expected format

        last_coma_index = response.rfind(",")
        if last_coma_index == response_length - 1:
            return json.dumps(
                {"questions": json.loads(response[:-1] + "]")}
            )  # Add missing closing bracket and remove the last coma

        # Remove the last incomplete question and add missing closing bracket
        return json.dumps({"questions": json.loads(response[:last_coma_index] + "]")})

    last_coma_index = response.rfind(",")
    if last_coma_index == response_length - 1:
        return response[:-1] + "]}"  # Add missing closing brackets and remove the last coma

    # Remove the last incomplete question and add missing closing brackets
    return response[:last_coma_index] + "]}"
