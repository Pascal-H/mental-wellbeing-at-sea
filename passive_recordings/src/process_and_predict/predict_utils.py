import os
import pickle
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr

import auvad
import audiofile as af
import audmodel
import audonnx
import audinterface
import audeer
import auglib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=[('CUDAExecutionProvider', {'device_id': 0})])


def get_dct_files(dir_in, lst_filter_mics):
    """
    Organize audio files by microphone channel into a dictionary structure.

    Parameters
    ----------
    dir_in : str
        Input directory containing FLAC audio files.
    lst_filter_mics : list
        List of microphone identifiers to filter by. If empty, includes all microphones.

    Returns
    -------
    dict
        Dictionary mapping microphone identifiers (M1, M2, M3, M6, V4, V5) to lists of filenames.
    """
    dct_files = {
        "M1": [],
        "M2": [],
        "M3": [],
        "V4": [],
        "V5": [],
        "M6": [],
    }
    keys_files = list(dct_files.keys())
    for file_name in os.listdir(dir_in):
        if file_name.endswith(".flac"):
            if lst_filter_mics:
                if any(string in file_name for string in lst_filter_mics):
                    cur_mic = next(s for s in keys_files if s in file_name)
                    dct_files[cur_mic].append(file_name)
            else:
                cur_mic = next(s for s in keys_files if s in file_name)
                dct_files[cur_mic].append(file_name)

    return dct_files


def find_quiet_segments(audio, sr, segment_len=5, segment_num=12):
    """
    Find quiet segments in audio for background noise estimation using VAD.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio signal array.
    sr : int
        Sample rate of the audio.
    segment_len : int, optional
        Length of each segment in seconds (default is 5).
    segment_num : int, optional
        Number of quiet segments to find (default is 12).

    Returns
    -------
    numpy.ndarray or None
        Concatenated background noise segments, or None if suitable segments not found.
    """
    # Initialise the VAD object
    v = auvad.Vad(
        min_turn_length=0.76, max_turn_length=5, num_workers=32, verbose=True  # 1.8,
    )
    # Split the audio into 5-second segments
    segment_length = segment_len * sr  # seconds in samples
    segments = [
        audio[i : i + segment_length] for i in range(0, len(audio), segment_length)
    ]

    # Find the segments with the lowest amplitude
    lowest_amp_segments = []
    rms_values = []
    for segment in segments:
        # Perform speech recognition on the segment
        vad_out = v.process_signal(segment, sr)
        # If speech is detected, skip the segment
        if not vad_out.empty:
            continue
        # Calculate the RMS amplitude of the segment
        rms = librosa.feature.rms(y=segment)[0][0]
        rms_values.append(rms)
        if len(lowest_amp_segments) < segment_num:
            lowest_amp_segments.append((segment, rms))
            lowest_amp_segments = sorted(lowest_amp_segments, key=lambda x: x[1])
        elif rms < lowest_amp_segments[-1][1]:
            lowest_amp_segments[-1] = (segment, rms)
            lowest_amp_segments = sorted(lowest_amp_segments, key=lambda x: x[1])

    # Concatenate the lowest amplitude segments to create a single background noise clip
    lowest_amp_segments = sorted(lowest_amp_segments, key=lambda x: x[1])
    background_noise_segments = [segment for segment, _ in lowest_amp_segments]
    background_noise = None
    try:
        background_noise = librosa.util.buf_to_float(
            np.concatenate(background_noise_segments)
        )
    except ValueError:
        print(f"Could not find suitable background noise!")
        background_noise_segments = None

    return background_noise


def denoise(denoising_approach, dir_in, dir_denoised, dct_files):
    """
    Apply denoising to audio files using specified denoising approach.

    Parameters
    ----------
    denoising_approach : str
        Denoising method to use ("noisereduce", "open_universe-plusplus", "facebook_denoiser", or "none").
    dir_in : str
        Input directory containing original audio files.
    dir_denoised : str
        Output directory for denoised audio files.
    dct_files : dict
        Dictionary mapping microphone channels to lists of filenames.
    """
    for cur_mic in list(dct_files.keys()):
        print(f"\nChecking for already denoised files for mic: {cur_mic}")
        for cur_file in audeer.progress_bar(dct_files[cur_mic]):
            # Check if denoising was already done by comparing the sampling rate and length
            denoised_path = os.path.join(dir_denoised, cur_file)
            raw_path = os.path.join(dir_in, cur_file)
            already_denoised = False

            # Check if the denoised file exists and is a .flac file
            if os.path.isfile(denoised_path) and denoised_path.endswith(".flac"):
                try:
                    # Read file info (length and sample rate)
                    raw_sig, raw_sr = af.read(raw_path)
                    denoised_sig, denoised_sr = af.read(denoised_path)
                    print(f"\n  Found denoised file: {cur_file}")
                    print(f"    Raw:       sr={raw_sr}, len={len(raw_sig)}")
                    print(f"    Denoised:  sr={denoised_sr}, len={len(denoised_sig)}")
                    if raw_sr == denoised_sr and len(raw_sig) == len(denoised_sig):
                        print(
                            "    ✔ Sampling rate and length match. Skipping denoising."
                        )
                        already_denoised = True
                    else:
                        print(
                            "    ✗ Sampling rate or length mismatch. Will re-denoise."
                        )
                except Exception as e:
                    print(
                        f"    ✗ Error reading files for comparison: {e}. Will re-denoise."
                    )
            else:
                raw_sig, raw_sr = af.read(raw_path)
            if already_denoised:
                continue
            print(f"Denoising {cur_file} for mic {cur_mic} with {denoising_approach}")

            # Denoising
            cur_sig = raw_sig
            cur_sr = raw_sr
            if denoising_approach == "no_denoising":
                # No denoising, just copy the file
                # (the actual file writing is done at the end of this conditional)
                cur_reduced_noise = cur_sig

            elif denoising_approach == "noisereduce":
                # Downsample files before denoising
                downsample = auglib.transform.Resample(
                    target_rate=16000, preserve_level=True
                )
                augment = auglib.Augment(downsample)
                # augment returns a series - get only the raw signal
                # Get also the first item, since we only expect 1 channel
                audio = augment.process_signal(cur_sig, cur_sr).values[0][0]
                background_noise = find_quiet_segments(audio, cur_sr)
                cur_reduced_noise = None
                if background_noise is not None:
                    cur_reduced_noise = nr.reduce_noise(
                        y=audio,
                        sr=cur_sr,
                        y_noise=background_noise,
                        stationary=False,
                        n_jobs=8,
                    )
                # Apply stationary noise reduction if no non-speech, noisy segments were found
                else:
                    cur_reduced_noise = nr.reduce_noise(
                        y=audio,
                        sr=cur_sr,
                        # y_noise=background_noise,
                        stationary=True,
                        n_jobs=8,
                    )
            elif denoising_approach == "open_universe-plusplus":
                import torch
                import torchaudio
                from open_universe import inference_utils

                # convert the current signal into a tensor
                if cur_sig.ndim == 1:
                    cur_sig = torch.from_numpy(cur_sig).unsqueeze(0)
                else:
                    raise ValueError(
                        f"Input signal has {cur_sig.ndim} dimensions, expected mono."
                    )

                # use GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # load enhancement model (from checkpoint file or HF repo)
                # Monkey patch needed: load_model() call torch.load(), which
                # in recent versions only supports loading of specific checkpoints
                # --> add `weights_only=False` in that function to load the whole model
                model = inference_utils.load_model(
                    "line-corporation/open-universe:plusplus", device=device
                )

                # Check if the proper sampling rate is used
                if cur_sr != model.fs:
                    raise ValueError(
                        f"Audio file and model do not use the same sampling frequency"
                    )

                # Need to chunk the audio due to GPU memory limitations
                chunk_duration = 240  # seconds; based on trial and error
                chunk_size = chunk_duration * cur_sr
                num_chunks = int(np.ceil(cur_sig.shape[1] / chunk_size))

                denoised_chunks = []
                with torch.no_grad():
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, cur_sig.shape[1])
                        chunk = cur_sig[:, start:end]
                        if chunk.shape[1] == 0:
                            continue
                        # The actual denoising
                        denoised_chunk = model.enhance(chunk.to(device))
                        denoised_chunks.append(denoised_chunk.cpu())

                # Concatenate the denoised chunks
                cur_reduced_noise_torch = torch.cat(denoised_chunks, dim=1)
                # Convert back to numpy array for processing with audiofile
                cur_reduced_noise = cur_reduced_noise_torch.cpu().numpy()

            elif denoising_approach == "facebook_denoiser":
                import torch

                # convert the current signal into a tensor
                if cur_sig.ndim == 1:
                    cur_sig = torch.from_numpy(cur_sig).unsqueeze(0)
                else:
                    raise ValueError(
                        f"Input signal has {cur_sig.ndim} dimensions, expected mono."
                    )

                # use GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load model ONCE
                from denoiser import pretrained

                model = pretrained.master64().to(device)
                model.eval()

                # # Need to chunk the audio due to GPU memory limitations
                # chunk_duration = 360  # seconds; based on trial and error
                # chunk_size = chunk_duration * cur_sr
                # num_chunks = int(np.ceil(cur_sig.shape[1] / chunk_size))

                # Ensure that the chunk size is a multiple of the stride
                # Had some issue before, where concatenating the chunks
                # did not work - this stride implementation is probably not
                # needed anymore, but let's keep it for now
                stride = model.stride  # Should be 4
                chunk_duration = 300  # Whatever fits in the GPU memory
                chunk_size = int(chunk_duration * cur_sr)
                # print(chunk_size)
                chunk_size = (
                    chunk_size // stride
                ) * stride  # Make it a multiple of stride
                # print(chunk_size)
                num_chunks = int(np.ceil(cur_sig.shape[1] / chunk_size))

                denoised_chunks = []
                total_samples = 0
                with torch.no_grad():
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, cur_sig.shape[1])
                        chunk = cur_sig[:, start:end]
                        if chunk.shape[1] == 0:
                            continue
                        # The actual denoising
                        # Take the result from the model and move it to CPU
                        denoised_chunk = model(chunk.to(device)).cpu()
                        # Remove batch dimension if present
                        if denoised_chunk.dim() == 3 and denoised_chunk.shape[0] == 1:
                            denoised_chunk = denoised_chunk.squeeze(0)
                        # Don't pad with zeros, just append the actual output
                        denoised_chunks.append(denoised_chunk)
                        total_samples += denoised_chunk.shape[1]
                        # print(
                        #     f"Chunk {i}: input {chunk.shape[1]}, output {denoised_chunk.shape[1]}"
                        # )

                # Concatenate all outputs
                cur_reduced_noise_torch = torch.cat(denoised_chunks, dim=1)
                cur_reduced_noise = cur_reduced_noise_torch.cpu().numpy()

            # Save the denoised and downsampled file
            af.write(os.path.join(dir_denoised, cur_file), cur_reduced_noise, cur_sr)

    return


def validate_and_fix_segments(df, root, shift_ms=50, max_attempts=5):
    """
    Validate and adjust audio segment indices to avoid read errors.

    This function iterates over the segments defined in the input DataFrame's MultiIndex,
    attempting to read each segment from its corresponding audio file. If a segment read
    fails due to a 'psf_fseek' error (typically caused by an invalid end time), the function
    incrementally increases the segment's end time by `shift_ms` milliseconds, retrying up
    to `max_attempts` times. Successfully validated (and possibly adjusted) segments are
    collected and returned as a new DataFrame with the updated MultiIndex.

    Args:
        df (pd.DataFrame): Input DataFrame with a MultiIndex of (file, start, end).
        root (str): Root directory containing the audio files.
        shift_ms (int, optional): Milliseconds to increment the segment end time on error. Default is 50.
        max_attempts (int, optional): Maximum number of attempts to fix a segment. Default is 5.

    Returns:
        pd.DataFrame: New DataFrame with a MultiIndex of validated (and possibly adjusted) segments.

    Notes:
        - Segments that cannot be validated after the specified number of attempts are skipped.
        - Only the MultiIndex is preserved in the returned DataFrame; no columns are included.
    """
    new_idx = []
    for idx in df.index:
        file, start, end = idx
        file_path = os.path.join(root, file)
        seg_start = pd.to_timedelta(start).total_seconds()
        seg_end = pd.to_timedelta(end).total_seconds()
        orig_end = seg_end
        success = False
        for attempt in range(max_attempts):
            try:
                _ = af.read(
                    file_path,
                    offset=seg_start,
                    duration=seg_end - seg_start,
                )
                success = True
                break
            except Exception as e:
                if "psf_fseek" in str(e):
                    seg_end += shift_ms / 1000.0
                else:
                    break
        if success and seg_end > seg_start:
            if seg_end != orig_end:
                new_end = pd.to_timedelta(seg_end, unit="s")
                print(
                    f"[validate_and_fix_segments]: Adjusting segment end for {file}: {start} - {end} to {new_end}"
                )
                idx = (file, start, new_end)
            new_idx.append(idx)
        else:
            print(
                f"[validate_and_fix_segments] Skipping segment: {file}, {start} - {end}"
            )
    # Return a new DataFrame with only the MultiIndex, no columns
    return pd.DataFrame(index=pd.MultiIndex.from_tuples(new_idx, names=df.index.names))


def vad_segments_exist(dir_segments, parent_folder, dct_files):
    """
    Check if VAD segment files exist and are non-empty for all microphones.

    Parameters
    ----------
    dir_segments : str
        Directory containing VAD segment files.
    parent_folder : str
        Name of the parent folder for constructing filenames.
    dct_files : dict
        Dictionary mapping microphone channels to lists of filenames.

    Returns
    -------
    bool
        True if all VAD segment files exist and are non-empty, False otherwise.
    """
    for mic in dct_files.keys():
        seg_path = os.path.join(dir_segments, f"segments_{parent_folder}_{mic}.pkl")
        if not os.path.isfile(seg_path):
            print(f"VAD segments file does not exist: {seg_path}")
            return False
        try:
            df = pd.read_pickle(seg_path)
            if len(df.index) == 0:
                print(f"VAD segments file is empty: {seg_path}")
                return False
        except Exception:
            print(f"Error reading VAD segments file: {seg_path}")
            return False
    return True


def vad(vad_approach, dir_in, dir_cache, dir_out, dct_files):
    """
    Perform Voice Activity Detection (VAD) on audio files using specified approach.

    Parameters
    ----------
    vad_approach : str
        VAD method to use ("auvad", "devaice", or pyannote model name).
    dir_in : str
        Input directory containing audio files.
    dir_cache : str
        Cache directory for temporary files.
    dir_out : str
        Output directory for VAD results.
    dct_files : dict
        Dictionary mapping microphone channels to lists of filenames.

    Returns
    -------
    dict
        Dictionary mapping microphone channels to DataFrames containing VAD segments.
    """
    parent_folder = os.path.basename(os.path.normpath(dir_in))
    audeer.mkdir(dir_out)
    dir_segments = os.path.join(dir_out, f"segments-{vad_approach}")
    audeer.mkdir(dir_segments)
    dct_df_segments = {}

    # --- Check if VAD has already been done ---
    if vad_segments_exist(dir_segments, parent_folder, dct_files):
        print("VAD segments already exist for all microphones. Loading from disk.")
        for mic in dct_files.keys():
            seg_path = os.path.join(dir_segments, f"segments_{parent_folder}_{mic}.pkl")
            dct_df_segments[mic] = pd.read_pickle(seg_path)
        return dct_df_segments

    # Always extract VAD segments for all microphones
    if "pyannote/" in vad_approach:
        import torch
        from pyannote.audio import Pipeline

        # Get the Hugging Face token from the environment variable
        token_hf = os.getenv("HUGGINGFACE_TOKEN")

        pyannote_model = Pipeline.from_pretrained(vad_approach, use_auth_token=token_hf)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pyannote_model.to(device)

        for cur_mic in list(dct_files.keys()):
            print(f"Segmenting {cur_mic}")
            # Initialize a dictionary to hold segments
            segments = {"file": [], "start": [], "end": [], "speaker": []}

            for cur_file in audeer.progress_bar(dct_files[cur_mic]):
                # Get the diarization for the current file
                cur_diarization = pyannote_model(os.path.join(dir_in, cur_file))
                # Iterate over the diarization turns and collect segments
                for turn, _, speaker in cur_diarization.itertracks(yield_label=True):
                    # TODO: use relative or absolute path?
                    segments["file"].append(os.path.join(dir_in, cur_file))
                    # The output is in seconds
                    # --> convert to timedelta
                    segments["start"].append(pd.to_timedelta(turn.start, unit="s"))
                    segments["end"].append(pd.to_timedelta(turn.end, unit="s"))
                    segments["speaker"].append(str(speaker))

            # Convert the segments dictionary to a DataFrame
            df_segments = pd.DataFrame(segments)
            # Set MultiIndex and drop the default index
            df_segments = df_segments.set_index(["file", "start", "end"], drop=True)
            # Due to very rate pfseek error: have to validate and fix the resulting segments index
            df_segments = validate_and_fix_segments(df_segments, dir_in)
            # Save the DataFrame to .csv
            dct_df_segments[cur_mic] = df_segments
            df_segments.to_pickle(
                os.path.join(dir_segments, f"segments_{parent_folder}_{cur_mic}.pkl")
            )

    elif vad_approach == "devaice":
        # If auvad is to be used and devAIce is not installed: avoid import error
        import devaice

        def apply_devaice_vad(
            signal,
            sampling_rate,
            processing_params,
            session,
            resource_root_path,
            serial_key,
        ):
            import devaice

            def get_bit_depth(signal):
                return signal.dtype.itemsize * 8

            # Create a raw audio format object for the audio signal
            raw_audio_format = devaice.DevaiceRawAudioFormat(
                sample_type=devaice.DEVAICE_SAMPLE_TYPE_FLOAT,
                bit_depth=get_bit_depth(signal),
                num_channels=1,
                sample_rate=sampling_rate,
            )
            # Initialise the VAD configuration and apply the values from the config
            vad_config = devaice.DEVAICE_VAD_CONFIG_DEFAULT()
            min_segment_length = processing_params.get("min_segment_length", None)
            if min_segment_length is not None:
                vad_config.min_segment_length = min_segment_length
            max_segment_length = processing_params.get("max_segment_length", None)
            if max_segment_length is not None:
                vad_config.max_segment_length = max_segment_length
            # vad_config.segment_start_delay = processing_params.get(
            #     "segment_start_delay", None
            # )
            # vad_config.segment_end_delay = processing_params.get(
            #     "segment_end_delay", None
            # )
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

        # Create a devAIce session
        session = devaice.MainDevaiceSession()
        # Set the path to the devAIce installation folder
        resource_root_path = (
            "/scratch/phecker/devAIce-SDK-3.12.0-audEERING-Research/res/"
        )
        # The devAIce serial key is stored in an environment variable to separate if from the source
        serial_key = os.getenv("DEVAICE_SERIAL_KEY")
        # Create an audinterface object for the VAD processing
        interface = audinterface.Segment(
            process_func=apply_devaice_vad,
            process_func_args={
                "processing_params": {
                    # default settings
                    "min_segment_length": 0.76,
                    "max_segment_length": 10.0,
                    # convert frames of 10ms into seconds
                    # segment_start_delay: 0.150 # 15 frames
                    # segment_end_delay: 0.25 # 25 frames
                },
                "session": session,
                "resource_root_path": resource_root_path,
                "serial_key": serial_key,
            },
            # Progress bar
            verbose=True,
            # num_workers seems to crash the code with
            # devaice.base_wrapper.DevaiceException: Result: 6, Message: Data has already been set for the stream
        )
        # # Create UID for cache directory based on the VAD settings
        # cache_vad = audeer.mkdir(
        #     cache_base
        #     / processing_type
        #     / audeer.uid(from_string=str(processing_params), short=True)
        # )
        for cur_mic in list(dct_files.keys()):
            print(f"Segmenting {cur_mic}")
            df_to_segment = pd.DataFrame(index=dct_files[cur_mic])
            # Process the index of the DataFrame
            idx_files_vad = interface.process_index(
                index=df_to_segment.index,
                root=dir_in,
                # cache_root=cache_vad,
            )

            dct_df_segments[cur_mic] = idx_files_vad
            idx_files_vad.to_pickle(
                os.path.join(dir_segments, f"segments_{parent_folder}_{cur_mic}.pkl")
            )

        session.free()

    elif vad_approach == "auvad":
        v = auvad.Vad(
            min_turn_length=0.76,  # 1.8,
            max_turn_length=10,
            num_workers=8,
            verbose=True,
        )
        for cur_mic in list(dct_files.keys()):
            print(f"Segmenting {cur_mic}")
            df_segments = pd.DataFrame()
            for cur_file in audeer.progress_bar(dct_files[cur_mic]):
                cur_sig, cur_fs = af.read(os.path.join(dir_in, cur_file))
                df_cur_segments = v.process_signal(cur_sig, cur_fs, file=cur_file)
                df_cur_segments = df_cur_segments.to_frame()
                df_cur_segments = df_cur_segments.drop(columns=["file", "start", "end"])
                # Due to very rate pfseek error: have to validate and fix the resulting segments index
                df_cur_segments = validate_and_fix_segments(df_cur_segments, dir_in)
                df_segments = pd.concat([df_segments, df_cur_segments])

            dct_df_segments[cur_mic] = df_segments
            df_segments.to_pickle(
                os.path.join(dir_segments, f"segments_{parent_folder}_{cur_mic}.pkl")
            )

    return dct_df_segments


def predict_emotion(
    dct_df_segments,
    parent_folder,
    dir_in,
    dir_cache,
    dir_out,
    denoising_specifier,
    vad_specifier,
    model_id,
    flag_transcribe,
):
    """
    Predict emotion dimensions (valence, arousal, dominance) from audio segments.

    Parameters
    ----------
    dct_df_segments : dict
        Dictionary mapping microphone channels to DataFrames containing VAD segments.
    parent_folder : str
        Name of the parent folder for organizing outputs.
    dir_in : str
        Input directory containing audio files.
    dir_cache : str
        Cache directory for temporary files.
    dir_out : str
        Output directory for prediction results.
    denoising_specifier : str
        Identifier for the denoising approach used.
    vad_specifier : str
        Identifier for the VAD approach used.
    model_id : str
        Identifier for the emotion prediction model.
    flag_transcribe : str
        Whether to include transcription ("true" or "false").

    Returns
    -------
    dict
        Dictionary mapping microphone channels to DataFrames containing emotion predictions.
    """
    dct_predictions = {}
    # Load the emotion model
    model_id = model_id
    model_root = audmodel.load(
        model_id,
        verbose=True,
    )
    model_onnx = audonnx.load(
        model_root, device="cuda"
    )  # [('CUDAExecutionProvider', {'device_id': 7})])
    model = audinterface.Process(process_func=model_onnx)

    # Label the output directory based on the model ID
    model_specifier = None
    if model_id == "497f50d6-1.1.0":
        model_specifier = "-mobilenet"

    if flag_transcribe == "true":
        model_specifier += "-transcripts"
    elif flag_transcribe == "false":
        model_specifier += "-no_transcripts"

    print(
        f"Using model ID: {model_id}, model_specifier: {model_specifier}, denoising: {denoising_specifier}, vad: {vad_specifier}, transcribe: {flag_transcribe}"
    )

    dir_predictions = os.path.join(
        dir_out,
        "predictions-emotion"
        + f"-{denoising_specifier}"
        + f"-{vad_specifier}"
        + model_specifier,
    )
    audeer.mkdir(dir_predictions)

    for cur_mic in list(dct_df_segments.keys()):
        print(f"Predicting emotion {cur_mic}")
        cur_df = dct_df_segments[cur_mic]
        if len(cur_df.index) == 0:
            print(f"[predict_utils] Empty DataFrame for {cur_mic}. Skipping.")
            continue
        predictions = model.process_index(cur_df.index, root=dir_in)

        # Process predictions
        predictions = predictions.to_frame()
        # Rename column called '0' with raw output
        predictions = predictions.rename(columns={0: "output_raw"})

        predictions["prediction_arousal"] = predictions["output_raw"].apply(
            lambda x: x["arousal"][0][0]
            # lambda x: x["logits"][0][0]
        )
        predictions["prediction_dominance"] = predictions["output_raw"].apply(
            lambda x: x["dominance"][0][0]
            # lambda x: x["logits"][0][1]
        )
        predictions["prediction_valence"] = predictions["output_raw"].apply(
            lambda x: x["valence"][0][0]
            # lambda x: x["logits"][0][2]
        )

        # Whisper transcription
        if flag_transcribe == "true":
            predictions = transcribe_whisper(
                str_model_size="turbo",
                device="cuda",
                df_predictions=predictions,
                dir_in=dir_in,
                min_seg_len=3,
            )

        # Add number column to have identifier for segment when later listening to it
        # predictions.insert(0, 'increment_id', [str(e).zfill(4) for e in list(range(0, len(predictions)))])
        dct_predictions[cur_mic] = predictions
        predictions.to_pickle(
            os.path.join(dir_predictions, f"predictions_{parent_folder}_{cur_mic}.pkl")
        )

    path_out_dict = os.path.join(
        dir_predictions, f"dct_predictions_{parent_folder}.pkl"
    )
    with open(path_out_dict, "wb") as f:
        pickle.dump(dct_predictions, f)

    return dct_predictions


def predict_snr(dct_df_segments, parent_folder, dir_denoised, dir_raw, dir_out):
    """
    Predict Signal-to-Noise Ratio (SNR) for audio segments.

    Parameters
    ----------
    dct_df_segments : dict
        Dictionary mapping microphone channels to DataFrames containing VAD segments.
    parent_folder : str
        Name of the parent folder for organizing outputs.
    dir_denoised : str
        Directory containing denoised audio files.
    dir_raw : str
        Directory containing raw audio files.
    dir_out : str
        Output directory for SNR prediction results.

    Returns
    -------
    dict
        Dictionary mapping microphone channels to DataFrames containing SNR predictions.
    """
    dct_predictions = {}
    # Load the SNR model
    model_root = audmodel.load(
        "b98a737c-2.4.1",
        verbose=True,
    )
    model_onnx = audonnx.load(model_root, device="cuda")
    model = audinterface.Process(process_func=model_onnx)

    dir_predictions = os.path.join(dir_out, "predictions-snr")
    audeer.mkdir(dir_predictions)

    # Get the labels for the logits output
    logits_labels = model_onnx.outputs[
        "logits"
    ].labels  # ['snr', 'reverberation-time', 'mos']

    for cur_mic in list(dct_df_segments.keys()):
        cur_df = dct_df_segments[cur_mic]
        if len(cur_df.index) == 0:
            print(f"[predict_utils] Empty DataFrame for {cur_mic}. Skipping.")
            continue

        # Iterate over the VAD segments
        # Predict SNR values for both denoised and raw audio files
        for root_type, root_dir in [("denoised", dir_denoised), ("raw", dir_raw)]:
            print(f"Predicting SNR {cur_mic} ({root_type})")
            predictions = model.process_index(cur_df.index, root=root_dir)
            predictions = predictions.to_frame()
            predictions = predictions.rename(columns={0: "output_raw"})

            # Map logits to columns using the labels
            for i, label in enumerate(logits_labels):
                predictions[label] = predictions["output_raw"].apply(
                    lambda x: x["logits"][0][i]
                )

            # Save the predictions
            key = f"{cur_mic}_{root_type}"
            dct_predictions[key] = predictions
            predictions.to_pickle(
                os.path.join(
                    dir_predictions,
                    f"predictions_{parent_folder}_{cur_mic}_{root_type}.pkl",
                )
            )

    path_out_dict = os.path.join(
        dir_predictions, f"dct_predictions_{parent_folder}.pkl"
    )
    with open(path_out_dict, "wb") as f:
        pickle.dump(dct_predictions, f)

    return dct_predictions


def transcribe_whisper(
    str_model_size="large",
    device="cuda",
    df_predictions=None,
    dir_in=None,
    min_seg_len=3,
):
    """
    Transcribe audio segments using OpenAI Whisper model.

    Parameters
    ----------
    str_model_size : str, optional
        Whisper model size to use (default is "large").
    device : str, optional
        Device to run inference on ("cuda" or "cpu", default is "cuda").
    df_predictions : pandas.DataFrame, optional
        DataFrame containing prediction data with audio segment information.
    dir_in : str, optional
        Input directory containing audio files.
    min_seg_len : int, optional
        Minimum segment length in seconds for transcription (default is 3).

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added "transcript" column containing transcription results.
    """
    import whisper
    import inspect
    import os

    model = whisper.load_model(str_model_size)
    model = model.to(device)

    print(f"Using the following device: {next(model.parameters()).device}")
    print(f"Current process ID: {os.getpid()}")

    sig = inspect.signature(whisper.log_mel_spectrogram)
    n_mels = sig.parameters["n_mels"].default
    if str_model_size in ["large"]:
        n_mels = 128

    transcripts = []
    for idx in audeer.progress_bar(df_predictions.index, desc="Transcribing segments"):
        file, start, end = idx
        cur_path = os.path.join(dir_in, file)
        start_sec = pd.to_timedelta(start).total_seconds()
        end_sec = pd.to_timedelta(end).total_seconds()
        # Check if the segment is too short
        if end_sec - start_sec <= min_seg_len:
            transcripts.append("")
            continue
        # Load the full audio
        audio = whisper.load_audio(cur_path)
        # Whisper expects 16000 Hz, so resample if needed
        if hasattr(model, "sample_rate"):
            sample_rate = model.sample_rate
        else:
            sample_rate = 16000
        # If not already at 16000 Hz, resample
        if audio.shape[0] != sample_rate:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=audio.shape[0], target_sr=sample_rate
            )
        # Slice the segment (audio is a numpy array, 1D)
        segment = audio[int(start_sec * sample_rate) : int(end_sec * sample_rate)]
        result = model.transcribe(segment)
        transcript = result.get("text", "") if isinstance(result, dict) else result
        transcripts.append(transcript)

    df_predictions = df_predictions.copy()
    df_predictions["transcript"] = transcripts

    return df_predictions
