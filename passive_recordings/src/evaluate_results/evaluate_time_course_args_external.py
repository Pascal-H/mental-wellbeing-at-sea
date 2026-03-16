import os


# Original paths (require original data, not publicly available)
# class Args:
#     dir_output_root = "/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/output"
#     dir_evaluated = "/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/evaluated"
#     preds_str = "emotion"
#     flag_concat = "false"
#     path_active = "/data/share/aisoundlab-mental_wellbeing_at_sea/data_mwas_processed-final_data/final_data-df_files.csv"
#     path_time_course_events = "/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/src/evaluate_results/evaluate_time_course_events.yaml"


# Synthetic data paths (for pipeline verification without original data)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
_SYNTH_PASSIVE = os.path.join(_REPO_ROOT, "synthetic_data", "passive")


class Args:
    dir_output_root = os.path.join(_SYNTH_PASSIVE, "data", "output")
    dir_evaluated = os.path.join(_SYNTH_PASSIVE, "data", "evaluated")
    # Separate directory for plot output so generated plots do not clutter
    # the dir_evaluated folder that also contains pickle data files.
    dir_plots = os.path.join(
        _SYNTH_PASSIVE, "data", "evaluated", "synthetic-time_course"
    )
    preds_str = "emotion"
    flag_concat = "false"
    path_active = os.path.join(
        _SYNTH_PASSIVE, "data", "evaluated", "synthetic_active_data.csv"
    )
    path_time_course_events = os.path.join(
        _SYNTH_PASSIVE, "src", "evaluate_results", "evaluate_time_course_events.yaml"
    )
