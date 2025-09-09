# For testing purposes mostly


class Args:
    dir_in = "/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/interim/IMO_9510682_2023-01-28T08-48-40"
    path_cache = "/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/cache_analysis"
    path_out = "/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/output"
    lst_filter_mics = [
        "M1",
        "M2",
        "M3",
        "V4",
        "V5",
        "M6",
    ]  # ["M1", "M2", "M3", "V4", "V5", "M6"]
    denoising_approach = "facebook_denoiser"  # "no_denoising"  # open_universe-plusplus # facebook_denoiser # noisereduce
    vad_approach = "auvad"  # "devaice"  # "pyannote/speaker-diarization-3.1"  # "auvad"  # "pyannote/voice-activity-detection"
    model_id = "497f50d6-1.1.0"
    flag_transcribe = "false"

    gpu = "1"
