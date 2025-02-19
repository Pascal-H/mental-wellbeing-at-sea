import os
import glob
import yaml
import pandas as pd
import re

import audeer


# Helper functions to deal with average results from multiple folds in the outer CV
def is_nested_dict(d):
    return any(isinstance(v, dict) for v in d.values())


def handle_nested_dict(d):
    for key in d.keys():
        if isinstance(d[key], dict) and "average" in d[key]:
            d[key] = d[key]["average"]
    return d


if __name__ == "__main__":
    # Paths to get the data sources
    path_exec = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "mwas"  # biogen-ms
    # Path to the root folder to hold all the results directories
    path_results_root = audeer.path(path_exec + f"/../results/{dataset_name}/modelling")

    ### Here you can refine the retrieval ###
    cur_target = "who_5_percentage_score_corrected"  # disorder_bin  # stress_current  # stress_work_tasks
    # Custom term that can be used to select a subset of the results
    # E.g. term: "eGeMAPSv02", name: "egemaps" will select all results that have "/eGeMAPSv02" in the path
    # Default for everything: "term": None, "name": "everything"
    # {"term": None, "name": "everything"}
    search_term = {
        "term": "*cohort-all*/**/*personalisation-none/loso/*",  # wav2vec2*  # cohort-survey-daily
        "name": "outer_cv_loso-surveys_all",  # wav2vec2-embeddings  # daily_survey
    }

    # Compose the output directory
    path_results_out = audeer.path(
        path_exec + f"/../results/{dataset_name}/composed/{search_term['name']}"
    )
    audeer.mkdir(path_results_out)

    df_all_results = pd.DataFrame()
    # Iterate over all folders and subfolder to fetch all included compiled results files
    path_glob = None
    if not search_term["term"]:
        path_glob = os.path.join(
            path_results_root,
            f"{cur_target}/**/results-compiled.yaml",
        )
    else:
        path_glob = os.path.join(
            path_results_root,
            f"{cur_target}/**/{search_term['term']}/**/results-compiled.yaml",
        )

    # Iterate over all matching results folders
    for summary_file in glob.iglob(
        path_glob,
        recursive=True,
    ):
        with open(summary_file, "r") as stream:
            summary_dict = yaml.safe_load(stream)
            # If the dictionary is nested (average across several folds), we want to extract the "average" key
            summary_dict = handle_nested_dict(summary_dict)
            cur_df = pd.DataFrame(summary_dict, index=[0])
            # cur_df["prompt"] = get_prompt_string(summary_file)
            cur_df["path"] = summary_file.replace(path_results_root, "")
            df_all_results = pd.concat([df_all_results, cur_df])

    df_all_results.to_csv(
        audeer.path(path_results_out, f"results-{cur_target}.csv"),
        index=False,
    )
