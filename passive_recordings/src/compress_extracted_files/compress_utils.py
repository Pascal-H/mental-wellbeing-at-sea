import os
import shutil
import sox
import zipfile

import audeer


def copy_convert_flac(path_in, path_interim):
    """
    Convert WAV files to FLAC format and copy other files to interim directory.

    Parameters
    ----------
    path_in : str
        Path to the input directory containing WAV files and other files.
    path_interim : str
        Path to the interim directory where FLAC files and copies will be stored.
    """
    for cur_file in audeer.progress_bar(os.listdir(path_in)):
        # Skip contained and empty 'opt' directory
        if os.path.isfile(os.path.join(path_in, cur_file)):
            # Convert .wav files into flac through sox
            if audeer.file_extension(cur_file) == "wav":
                cur_basename = audeer.basename_wo_ext(cur_file)
                sox.core.sox(
                    [
                        os.path.join(path_in, cur_file),
                        os.path.join(path_interim, f"{cur_basename}.flac"),
                    ]
                )
            # Just copy all other files
            else:
                shutil.copy2(
                    os.path.join(path_in, cur_file),
                    os.path.join(path_interim, cur_file),
                )
    return


def archive_dir(path_interim, path_out, parent_folder):
    """
    Create a ZIP archive from the interim directory.

    Parameters
    ----------
    path_interim : str
        Path to the interim directory containing files to archive.
    path_out : str
        Path to the output directory where the ZIP file will be created.
    parent_folder : str
        Name of the parent folder, used as the archive filename.
    """
    shutil.make_archive(os.path.join(path_out, parent_folder), "zip", path_interim)
    return


def verify_interim(path_in, path_interim):
    """
    Verify that interim directory contents match original directory (accounting for WAV to FLAC conversion).

    Parameters
    ----------
    path_in : str
        Path to the original input directory.
    path_interim : str
        Path to the interim directory to verify.

    Raises
    ------
    Exception
        If there's a mismatch between original and interim file basenames.
    """
    # List of original files
    lst_in = os.listdir(path_in)
    # Use basename, since .wav files were converted to .flac files
    lst_in_basename = [audeer.basename_wo_ext(e) for e in lst_in]
    # List of interim files
    lst_interim = os.listdir(path_interim)
    lst_interim_basename = [audeer.basename_wo_ext(e) for e in lst_interim]

    # Forming sets for validation
    lst_interim_in = list(set(lst_interim_basename) - set(lst_in_basename))
    if lst_interim_in:
        raise Exception(
            f"Mismatch between original and interim files: {lst_interim_in}."
        )

    print("Checked interim directory contents: OK.")
    return


def verify_zip(path_in, path_interim, path_out, parent_folder):
    """
    Verify the integrity and contents of the created ZIP archive.

    Parameters
    ----------
    path_in : str
        Path to the original input directory.
    path_interim : str
        Path to the interim directory.
    path_out : str
        Path to the output directory containing the ZIP file.
    parent_folder : str
        Name of the parent folder, used as the archive filename.

    Raises
    ------
    Exception
        If the ZIP file is corrupted or contents don't match expected files.
    """
    # Check if the output .zip file is corrupted
    path_archive = os.path.join(path_out, f"{parent_folder}.zip")
    with zipfile.ZipFile(path_archive, mode="r") as archive:
        zip_corrupt = archive.testzip()
        if zip_corrupt:
            raise Exception(f"Corrupted .zip file: {zip_corrupt}")
        print("Checked archive: OK.")
        lst_zip = archive.namelist()

    ## Check if the file list of the original dir is matching the zipped on
    # Create version of list with basenames only
    lst_zip_basename = [audeer.basename_wo_ext(e) for e in lst_zip]
    # List of original files
    lst_in = os.listdir(path_in)
    lst_in_basename = [audeer.basename_wo_ext(e) for e in lst_in]
    # List of interim files
    lst_interim = os.listdir(path_interim)

    # Forming sets for validation
    lst_zip_in = list(set(lst_zip_basename) - set(lst_in_basename))
    if lst_zip_in:
        raise Exception(f"Mismatch between original and zipped files: {lst_zip_in}.")
    lst_zip_interim = list(set(lst_zip) - set(lst_interim))
    if lst_zip_interim:
        raise Exception(
            f"Mismatch between original and zipped files: {lst_zip_interim}."
        )

    print("Checked directory contents: OK.")
    return


def clean_up(path_in, path_interim=None):
    """
    Clean up temporary directories after successful processing.

    Parameters
    ----------
    path_in : str
        Path to the original input directory to remove.
    path_interim : str, optional
        Path to the interim directory to remove (default is None).
    """
    # Delete the original directory
    audeer.rmdir(path_in)
    print("Removed parent directory.")
    # Only delete interim directory, if it's path is given
    if path_interim:
        audeer.rmdir(path_interim)
        print("Removed interim directory.")
    return
