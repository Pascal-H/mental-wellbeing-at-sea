import os
import argparse

import audeer

import compress_utils
from compress_args_external import Args


def main(args):
    """
    Main function to orchestrate WAV to FLAC conversion and optional archiving.

    Parameters
    ----------
    args : Args
        Arguments object containing configuration parameters:
        - dir_in: Input directory with WAV files
        - path_interim: Interim directory for FLAC files
        - path_out: Output directory for ZIP archive (optional)
    """
    # Ensure proper paths
    args.dir_in = audeer.path(args.dir_in)
    args.path_interim = audeer.path(args.path_interim)
    # Path out is optional: if not given, only compress .wav to .flac into
    # interim directory
    if args.path_out:
        args.path_out = audeer.path(args.path_out)

    # Get name of uppermost directory: the folder that contains the extracted
    # files
    parent_folder = os.path.basename(os.path.normpath(args.dir_in))
    # Create the directories for the interim directory
    dir_interim = audeer.path(args.path_interim, parent_folder)
    # Remove interim directory if it exists
    if os.path.isdir(dir_interim):
        print(f'Found interim directory for "{parent_folder}" - deleting.')
        audeer.rmdir(dir_interim)
    audeer.mkdir(dir_interim)
    # Check if the archive already exists and delete it if it does
    if args.path_out:
        out_zip = os.path.join(args.path_out, f"{parent_folder}.zip")
        if os.path.isfile(out_zip):
            print(f'Found archive "{out_zip}" - deleting.')
            os.remove(out_zip)

    print("Converting .wav to .flac.")
    compress_utils.copy_convert_flac(args.dir_in, dir_interim)
    # Verify if files in interim folder match the ones in the base directory
    print("Verifying interim files.")
    compress_utils.verify_interim(args.dir_in, dir_interim)
    # Only if output path for .zip archive is given: archive and verify interim
    if args.path_out:
        print("Archiving .flac files.")
        compress_utils.archive_dir(dir_interim, args.path_out, parent_folder)
        print("Verifying .zip archive.")
        compress_utils.verify_zip(
            args.dir_in, dir_interim, args.path_out, parent_folder
        )
    print("Cleaning up.")
    compress_utils.clean_up(args.dir_in)
    print("Finished.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_in",
        metavar="dir_in",
        type=str,
        help=(
            "Path of the folder to convert the extracted .wav files into "
            ".flac and the remaining files for archiving."
        ),
    )
    parser.add_argument(
        "path_interim",
        metavar="path_interim",
        type=str,
        help=(
            "Path to folder where to to create a folder named like dir_in to "
            "temporarily store compressed .flac files."
        ),
    )
    parser.add_argument(
        "--path-out",
        dest="path_out",
        type=str,
        default=None,
        # required=False,
        help=(
            "Path to folder where to store the packed archive. "
            "If not given, .wav files will be converted to .flac in interim "
            "directory without archiving to .zip."
        ),
    )

    try:
        args = parser.parse_args()
    except:
        args = Args()

    main(args)
