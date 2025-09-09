import os
import argparse

import audeer

from predict_args_external import Args


def main(args):
    """
    Main function to orchestrate audio processing and emotion prediction pipeline.

    This function processes VDR audio data through denoising, voice activity detection (VAD),
    and emotion prediction stages. It handles multiple microphone channels and creates
    organized output directories for caching and results.

    Parameters
    ----------
    args : Args
        Arguments object containing configuration parameters:
        - dir_in: Input directory with audio files
        - path_cache: Cache directory for temporary files
        - path_out: Output directory for results
        - lst_filter_mics: List of microphones to process
        - denoising_approach: Denoising method to use
        - vad_approach: VAD method to use
        - model_id: Emotion prediction model identifier
        - flag_transcribe: Whether to include transcription
        - gpu: GPU identifier for processing
    """
    # Have to set the environment variables for CUDA before importing any
    # libraries that use it, such as torch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import predict_utils

    args.dir_in = audeer.path(args.dir_in)
    args.path_cache = audeer.path(args.path_cache)
    args.path_out = audeer.path(args.path_out)
    # args.lst_filter_mics

    # Get name of uppermost directory: the folder that contains the extracted
    # files
    parent_folder = os.path.basename(os.path.normpath(args.dir_in))
    print(f"Processing {parent_folder}.")
    # Create the folder in the cache directory
    dir_cache = audeer.path(args.path_cache, parent_folder)
    # Remove the contents of the cache directory if exists
    if os.path.isdir(dir_cache):
        print(f'Found cache directory for "{parent_folder}" - deleting.')
        audeer.rmdir(dir_cache)
    audeer.mkdir(dir_cache)
    # Directory to store the 'raw' denoised files in
    dir_denoised = audeer.path(
        args.path_cache, f"{parent_folder}-{args.denoising_approach}"
    )
    audeer.mkdir(dir_denoised)
    # Directory to store the cached denoised segments in
    dir_denoised_cache = audeer.path(
        args.path_cache, f"{parent_folder}-{args.denoising_approach}-cache"
    )
    audeer.mkdir(dir_denoised_cache)
    # Create the folder in the output directory
    dir_out = audeer.path(args.path_out, parent_folder)
    audeer.mkdir(dir_out)

    # Default: include all M and V microphone channels
    dct_files = predict_utils.get_dct_files(args.dir_in, args.lst_filter_mics)
    predict_utils.denoise(args.denoising_approach, args.dir_in, dir_denoised, dct_files)

    # change dir in to point at denoised files
    # dct_df_segments = predict_utils.vad(args.dir_in, dir_cache, dir_out, dct_files)

    # Denoised
    dct_df_segments = predict_utils.vad(
        args.vad_approach, dir_denoised, dir_denoised_cache, dir_out, dct_files
    )
    # Non-denoised segmentation
    # dct_df_segments = predict_utils.vad(
    #     args.vad_approach, args.dir_in, dir_cache, dir_out, dct_files
    # )

    # predict_utils.predict_snr(
    #     dct_df_segments, parent_folder, dir_denoised, args.dir_in, dir_out
    # )

    dct_predictions = predict_utils.predict_emotion(
        dct_df_segments,
        parent_folder,
        dir_denoised,
        dir_denoised_cache,
        dir_out,
        args.denoising_approach,
        args.vad_approach,
        args.model_id,
        args.flag_transcribe,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_in",
        metavar="dir_in",
        type=str,
        help=(
            "Path of the folder where the extracted and converted .flac files "
            "reside."
        ),
    )
    parser.add_argument(
        "path_cache",
        metavar="path_cache",
        type=str,
        help=(
            "Path to folder where to cache extracted VAD segments and other "
            "files temporarily."
        ),
    )
    parser.add_argument(
        "path_out",
        metavar="path_out",
        type=str,
        help=(
            "Path to folder where to store the resulting predictions "
            "DataFrames, plots and other materials."
        ),
    )
    parser.add_argument(
        "--lst_filter_mics",
        dest="lst_filter_mics",
        type=str,
        default=[],
        # required=False,
        help=(
            "If only specific microphones are to be processed, their "
            "identifiers can be given here. Valid microphones are: M1, M2, "
            "M3, M6, V4, V5 (V being the radio communication channels). "
            "Otherwise, all Microphone channels will be processed and saved "
            "separately."
        ),
    )
    parser.add_argument(
        "denoising_approach",
        metavar="denoising_approach",
        default="noisereduce",
        type=str,
        help=(
            "Experiment with different denoising strategies. "
            "The defaults is the Python package noisereduce. "
            "open-universe-plusplus: https://github.com/line/open-universe"
        ),
    )
    parser.add_argument(
        "vad_approach",
        metavar="vad_approach",
        default="auvad",
        type=str,
        help=(
            "Experiment with different vad strategies. "
            "The defaults is the audEERING-internal auvad. "
        ),
    )
    parser.add_argument(
        "model_id",
        metavar="model_id",
        default="6bc4a7fd-1.1.0",
        type=str,
        help=(
            "Experiment with different vad strategies. "
            "The defaults is the audEERING-internal auvad. "
        ),
    )
    parser.add_argument(
        "flag_transcribe",
        metavar="flag_transcribe",
        default="false",
        type=str,
        help=(
            "Experiment with different vad strategies. "
            "The defaults is the audEERING-internal auvad. "
        ),
    )
    parser.add_argument(
        "gpu",
        metavar="gpu",
        default="cpu",
        type=str,
        help=("Pass an identifier for the GPU to use, e.g. 'cuda:0' or 'cpu'."),
    )

    try:
        args = parser.parse_args()
    except:
        args = Args()

    main(args)
