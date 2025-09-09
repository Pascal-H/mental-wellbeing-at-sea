import os
import argparse

import audeer

import yaml

from evaluate_time_course_args_external import Args
import evaluate_time_course_utils
import evaluate_utils_snr_distribution


def main(args):
    """
    Main function to orchestrate time course evaluation and plotting.

    This function loads and processes both passive (VDR audio) and active (survey) data,
    generates SNR distribution plots, and creates time series visualizations with event markers.

    Parameters
    ----------
    args : Args
        Arguments object containing configuration parameters:
        - dir_output_root: Root directory with prediction outputs
        - dir_evaluated: Directory for evaluation results
        - flag_concat: Whether to concatenate predictions anew
        - preds_str: Prediction type identifier (e.g., "emotion")
        - path_active: Path to active survey data
        - path_time_course_events: Path to event timing YAML file
    """
    df_snr = evaluate_time_course_utils.concat_all_preds(
        args.dir_output_root, args.dir_evaluated, args.flag_concat, "snr"
    )

    # Filter the SNR DataFrame to only include segments with a minimum duration of 3 seconds
    df_snr_3s = evaluate_time_course_utils.filter_duration(df_snr, min_duration=3)
    # Aggregate over the individual time segments (--> ~3h long chunks)
    df_snr_3s = (
        df_snr_3s.groupby(["noise_status", "microphone", "time"])
        .mean(numeric_only=True)
        .reset_index()
    )

    out_dir_snr = os.path.join(args.dir_evaluated, "plots-snr-distribution-min3s")
    out_dir_snr = audeer.mkdir(out_dir_snr)
    evaluate_utils_snr_distribution.plot_snr_distribution(
        df_snr_3s, "M", out_dir_snr, "M"
    )
    evaluate_utils_snr_distribution.plot_snr_distribution(
        df_snr_3s, "V", out_dir_snr, "V"
    )

    df_passive = evaluate_time_course_utils.concat_all_preds(
        args.dir_output_root, args.dir_evaluated, args.flag_concat, args.preds_str
    )

    df_passive = evaluate_time_course_utils.filter_duration(df_passive, min_duration=3)
    # Aggregate over the individual time segments
    df_passive = (
        df_passive.groupby(["microphone", "time"]).mean(numeric_only=True).reset_index()
    )

    df_active = evaluate_time_course_utils.load_database_aisl(args.path_active)

    evaluate_time_course_utils.evaluate_time_course(
        df_passive=df_passive,
        df_active=df_active,
        path_time_course_events=args.path_time_course_events,
        dir_evaluated=args.dir_evaluated,
        preds_str=args.preds_str,
    )

    # Load the time course events from a YAML file
    with open(args.path_time_course_events, "r") as f:
        event_timings = yaml.safe_load(f)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_output_root",
        metavar="dir_output_root",
        type=str,
        help=(
            "Path of the folder where the output of the process_and_predict "
            "step is being stored."
        ),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_evaluated",
        metavar="dir_evaluated",
        type=str,
        help=(
            "Path of the folder where the results of the evaluation here is "
            "to be stored."
        ),
    )
    parser.add_argument(
        "preds_str",
        metavar="preds_str",
        default="emotion",
        type=str,
        help=(
            "Specifier string for the sub-directories in dir_output. "
            "E.g., the directories containing the emotion predictions are called "
            "'predictions-emotion' (--> the identifier is then just 'emotion')."
        ),
    )
    parser.add_argument(
        "flag_concat",
        metavar="flag_concat",
        default="true",
        type=str,
        help=(
            "Indicate if the individual prediction DataFrame should be "
            "concatenated anew or if they should be loaded from disk."
        ),
    )
    parser.add_argument(
        "path_active",
        metavar="path_active",
        default="true",
        type=str,
        help=(
            "Path to the metadata DataFrame of the actively collected "
            "(AI SoundLab-based) data."
        ),
    )
    parser.add_argument(
        "path_time_course_events",
        metavar="path_time_course_events",
        default="true",
        type=str,
        help=(
            "Path to the metadata YAML with the time stamps of the events during "
            "the ship's journey."
        ),
    )

    try:
        args = parser.parse_args()
    except:
        args = Args()

    main(args)
