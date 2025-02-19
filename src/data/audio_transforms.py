import audeer
import audinterface
import auglib

import os
import pandas as pd


def config_to_transforms(parameters_dict):
    """
    Converts a dictionary of parameters into a list of audio transforms.

    Args:
        parameters_dict (dict): A dictionary containing the parameters for potentially several transforms to be applied to an audio signal.

    Returns:
        auglib.transform.Compose: A composed audio transform object.

    Raises:
        ValueError: If an unknown transform or method is encountered.
    """
    transforms = []
    for processing_type, processing_params in parameters_dict.items():
        for augmentation_type, augmentation_params in processing_params.items():
            if processing_type == "denoising":
                if augmentation_type == "no_denoising":
                    continue
                else:
                    print(f"USING PROVIDED DENOISED FILES: {augmentation_type}")
                    continue
            elif processing_type == "loudness_normalization":
                if augmentation_type == "no_loudness_normalization":
                    continue
                elif augmentation_type == "LUFS":
                    transforms.append(
                        auglib.transform.Function(
                            apply_lufs_normalization, function_args=augmentation_params
                        )
                    )
                elif augmentation_type == "RMS":
                    transforms.append()
                elif augmentation_type == "DRC":
                    transforms.append()
                else:
                    raise ValueError(
                        f"Unknown loudness normalization method: {augmentation_type}"
                    )
            elif processing_type == "augmentation":
                # Add code to handle augmentation here
                if augmentation_type == "no_augmentation":
                    continue
                elif augmentation_type == "add_background_noise":
                    transforms.append()
                else:
                    raise ValueError(
                        f"Unknown augmentation method: {augmentation_type}"
                    )
            else:
                raise ValueError(f"Unknown transform: {processing_type}")
    return auglib.transform.Compose(transforms)


def apply_lufs_normalization(audio_data, sample_rate, target_lufs):
    """
    Apply loudness normalization to the audio data.
    Supports both mono and stereo audio data.

    Args:
        audio_data (numpy.ndarray): The audio data to be normalized.
        sample_rate (int): The sample rate of the audio data.
        target_lufs (float): The target loudness level in LUFS (Loudness Units Full Scale).

    Returns:
        numpy.ndarray: The normalized audio data.

    Raises:
        ValueError: If the number of audio channels is unsupported.
    """
    # Need direct imports here for auglib to work
    import numpy as np
    import pyloudnorm as pyln
    import warnings

    # Create a Meter instance
    meter = pyln.Meter(sample_rate)

    # Calculate the length of the audio data in seconds
    audio_length_seconds = audio_data.shape[0] / sample_rate

    # If the audio data is shorter than the block size, return it as is
    if audio_length_seconds < meter.block_size:
        warnings.warn(
            f"Audio data is shorter than the block size ({meter.block_size}). "
            f"Returning the audio data as is."
        )
        return audio_data

    # If the audio data is stereo, process each channel separately
    if audio_data.shape[0] == 2:
        # Calculate the loudness of each channel
        loudness_0 = meter.integrated_loudness(audio_data[0])
        loudness_1 = meter.integrated_loudness(audio_data[1])

        # Normalize each channel to the target LUFS
        normalized_audio_0 = pyln.normalize.loudness(
            audio_data[0], loudness_0, target_lufs
        )
        normalized_audio_1 = pyln.normalize.loudness(
            audio_data[1], loudness_1, target_lufs
        )

        # Combine the normalized channels into a single numpy array
        normalized_audio = np.array([normalized_audio_0, normalized_audio_1])
    # If the audio data is mono, process the single channel
    elif audio_data.shape[0] == 1:
        # Calculate the loudness of the audio data
        loudness = meter.integrated_loudness(audio_data[0])

        # Normalize the audio data to the target LUFS
        normalized_audio = pyln.normalize.loudness(audio_data[0], loudness, target_lufs)
    else:
        raise ValueError(f"Unsupported number of channels: {audio_data.shape[0]}")

    return normalized_audio


def apply_devaice_vad(
    signal, sampling_rate, processing_params, session, resource_root_path, serial_key
):
    """
    Apply the devAIce Voice Activity Detection (VAD) module on an audio signal.

    Parameters:
    signal (numpy.ndarray): The audio signal to process. Should be a 1D numpy array.
    sampling_rate (int): The sampling rate of the audio signal.
    session (devaice.Session): An instance of the devAIce session to use for VAD processing.
    resource_root_path (str): The root path to the devAIce installation.
    serial_key (str): The serial key for the devAIce installation.
    processing_params (dict): A dictionary containing VAD processing parameters:
        - "min_segment_length" (float): Minimum segment length in seconds.
        - "max_segment_length" (float): Maximum segment length in seconds.
        - "segment_start_delay" (float): Segment start delay in seconds.
        - "segment_end_delay" (float): Segment end delay in seconds.

    Returns:
    pd.MultiIndex: A MultiIndex with 'start' and 'end' levels representing the start and end times of detected speech segments.
    """

    def get_bit_depth(signal):
        return signal.dtype.itemsize * 8

    # If auvad is to be used and devAIce is not installed: avoid import error
    import devaice

    # Create a raw audio format object for the audio signal
    raw_audio_format = devaice.DevaiceRawAudioFormat(
        sample_type=devaice.DEVAICE_SAMPLE_TYPE_FLOAT,
        bit_depth=get_bit_depth(signal),
        num_channels=1,
        sample_rate=sampling_rate,
    )
    # Initialise the VAD configuration and apply the values from the config
    vad_config = devaice.DEVAICE_VAD_CONFIG_DEFAULT()
    vad_config.min_segment_length = processing_params.get("min_segment_length", None)
    vad_config.max_segment_length = processing_params.get("max_segment_length", None)
    vad_config.segment_start_delay = processing_params.get("segment_start_delay", None)
    vad_config.segment_end_delay = processing_params.get("segment_end_delay", None)
    # Perform VAD on the audio signal
    session_result = session.util_vad_segments_on_buffer(
        signal,
        vad_config=vad_config,
        raw_audio_format=raw_audio_format,
        resource_root_path=resource_root_path,
        serial_key=serial_key,
    )

    # Extract the start and end times of the detected speech segments
    starts = [segment.start for segment in session_result]
    ends = [segment.end for segment in session_result]

    # Convert the start and end times to pandas Timedeltas
    starts = pd.to_timedelta(starts, unit="s")
    ends = pd.to_timedelta(ends, unit="s")

    # Return a MultiIndex with 'start' and 'end' levels
    # This will be processed by audinterface and mapped to the "file" level
    return pd.MultiIndex.from_arrays([starts, ends], names=["start", "end"])


def apply_vad(df_files, processing_type, processing_params, cache_base):
    """
    Apply Voice Activity Detection (VAD) to the audio files in the given DataFrame.

    Args:
        df_files (pandas.DataFrame): DataFrame containing information about the audio files.
        processing_type (str): Type of VAD processing to apply. E.g., "no_vad" and "auvad".
        processing_params (dict): Parameters for the VAD processing. The specific parameters depend on the VAD method.
        cache_base (str): Base directory for caching the segmented DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the VAD-processed audio files.

    Raises:
        ValueError: If the specified VAD method is unknown.
    """
    # TODO: num_workers
    if processing_type == "no_vad":
        df_files_vad = df_files

    elif processing_type == "devaice_vad":
        # If auvad is to be used and devAIce is not installed: avoid import error
        import devaice

        # Create a devAIce session
        session = devaice.MainDevaiceSession()
        # Set the path to the devAIce installation folder
        resource_root_path = "/data/phecker/devAIce-SDK-3.12.0-audEERING-Research/res/"
        # The devAIce serial key is stored in an environment variable to separate if from the source
        serial_key = os.getenv("DEVAICE_SERIAL_KEY")
        # Create an audinterface object for the VAD processing
        interface = audinterface.Segment(
            process_func=apply_devaice_vad,
            process_func_args={
                "processing_params": processing_params,
                "session": session,
                "resource_root_path": resource_root_path,
                "serial_key": serial_key,
            },
            # Progress bar
            verbose=True,
            # num_workers seems to crash the code with
            # devaice.base_wrapper.DevaiceException: Result: 6, Message: Data has already been set for the stream
        )
        # Create UID for cache directory based on the VAD settings
        cache_vad = audeer.mkdir(
            cache_base
            / processing_type
            / audeer.uid(from_string=str(processing_params), short=True)
        )
        # Process the index of the DataFrame
        idx_files_vad = interface.process_index(
            index=df_files.index,
            cache_root=cache_vad,
        )
        session.free()
        # Map the index to the labels
        df_files_vad = map_index_to_labels(df_files, idx_files_vad)

    elif processing_type == "auvad":
        # If devAIce is not installes and auvad is to be used: avoid import error
        import auvad

        vad = auvad.Vad(
            min_turn_length=processing_params.get("min_turn_length", None),
            max_turn_length=processing_params.get("max_turn_length", None),
            n_pre=processing_params.get("n_pre", None),
            n_post=processing_params.get("n_post", None),
            invert=processing_params.get("invert", None),
        )
        interface = audinterface.Segment(
            process_func=vad,
            num_workers=7,
            # num_workers=processing_params.get("num_workers", None),
            # Progress bar
            verbose=True,
        )
        # Create UID for cache directory based on the VAD settings
        cache_vad = audeer.mkdir(
            cache_base
            / processing_type
            / audeer.uid(from_string=str(processing_params), short=True)
        )
        idx_files_vad = interface.process_index(
            df_files.index,
            cache_root=cache_vad,
        )
        df_files_vad = map_index_to_labels(df_files, idx_files_vad)

    else:
        raise ValueError(f"Unknown VAD method: {processing_type}")

    return df_files_vad


def _convert_and_set_index(df):
    df["file"] = df["file"].astype(str)
    df["start"] = df["start"].apply(pd.to_timedelta)
    df["end"] = df["end"].apply(pd.to_timedelta)
    df.set_index(["file", "start", "end"], inplace=True)
    return df


def map_index_to_labels(df_labels, idx_segmented):
    """
    Maps the labels DataFrame to the segmented index by assigning the files to the segments index.

    Args:
        df_labels (pandas.DataFrame): The DataFrame containing the labels.
        idx_segmented (pandas.MultiIndex): The segmented index.

    Returns:
        pandas.DataFrame: The DataFrame with the mapped labels.
    """

    # Reset the index of df_labels so that the file names become a column
    df_labels_reset = df_labels.reset_index()

    # Drop the 'start' and 'end' columns
    df_labels_reset = df_labels_reset.drop(columns=["start", "end"])

    # Rename the index column to match the name in idx_segmented
    df_labels_reset = df_labels_reset.rename(columns={"index": "file"})

    # Convert idx_segmented to a DataFrame and reset the index
    df_segmented = pd.DataFrame(index=idx_segmented).reset_index()

    # Merge df_labels_reset with df_segmented on the file names
    df_merged = pd.merge(df_segmented, df_labels_reset, on="file")

    # Convert 'start' and 'end' to timedelta and set index for both df_merged and idx_segmented
    df_merged = _convert_and_set_index(df_merged)
    idx_segmented = _convert_and_set_index(df_segmented)

    # Assert that the index of df_merged is equal to idx_segmented
    assert df_merged.index.equals(
        idx_segmented.index
    ), "The index of df_merged is not equal to idx_segmented"

    # Assert that there are no duplicates in df_merged
    assert not df_merged.index.duplicated().any(), "There are duplicates in df_merged"

    return df_merged


def apply_transforms(df_files, transforms, cache_root=auglib.default_cache_root()):
    """
    Apply a list of audio transforms to a DataFrame of audio files.

    Args:
        df_files (pandas.DataFrame): DataFrame containing audio files and their labels.
        transforms (list): List of audio transforms to apply.
        cache_root (str, optional): Root directory for caching transformed audio files. Defaults to auglib's default_cache_root().

    Returns:
        pandas.DataFrame: DataFrame containing the paths to the augmented audio files.
    """
    if not transforms:
        df_files_augmented = df_files
    else:
        # TODO: num_workers
        augment = auglib.Augment(transforms)
        df_files_augmented = augment.augment(
            df_files,
            cache_root=cache_root,
        )

    return df_files_augmented
