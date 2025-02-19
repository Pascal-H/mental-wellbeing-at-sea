import gc
import re
import opensmile
import audeer
import audinterface
import pandas as pd
import torch
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    AutoConfig,
    Wav2Vec2FeatureExtractor,
)


class Wav2vec2:
    """
    Wav2vec2 class for extracting features from audio signals using the Wav2Vec2 model.

    Args:
        extract_options (dict): A dictionary containing the extraction options.
            - device (str, optional): The device to use for processing (default: "cuda" if available, else "cpu").
            - variant (str): The variant of the Wav2Vec2 model to use.
            - num_hidden_layers (int, optional): The number of hidden layers to exclude from the model (default: 0).

    Attributes:
        feature_set (str): The variant of the Wav2Vec2 model.
        extract_options (dict): The extraction options.
        device (str): The device used for processing.
        variant (str): The variant of the Wav2Vec2 model. Derived from feature_set (str).
        num_hidden_layers_to_exclude (int): The number of hidden layers to exclude from the model.
        model_initialized (bool): Indicates whether the model has been initialized.
        processor (Wav2Vec2FeatureExtractor): The feature extractor.
        model (Wav2Vec2Model): The Wav2Vec2 model.

    Methods:
        init_model(): Initializes the Wav2Vec2 model.
        get_hidden_size(): Returns the hidden size of the model.
        process_signal(signal, sampling_rate): Processes an audio signal and returns the extracted features.

    """

    def __init__(self, extract_options):
        """
        Initializes the Wav2Vec2 model by loading the pretrained model and configuring the processor.
        """

        self.extract_options = extract_options
        self.device = extract_options.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.path_suffix = extract_options["path_suffix"]
        self.variant = extract_options["variant"]
        self.num_hidden_layers_to_exclude = extract_options.get("num_hidden_layers", 0)
        self.model_initialized = False
        self.init_model()

    def init_model(self):
        model_path = f"{self.path_suffix}/{self.variant}"
        config = AutoConfig.from_pretrained(model_path)
        if self.num_hidden_layers_to_exclude > 0:
            config.num_hidden_layers -= self.num_hidden_layers_to_exclude
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path, config=config).to(
            self.device
        )
        self.model.eval()
        self.model_initialized = True

    def get_hidden_size(self):
        """
        Returns the hidden size of the Wav2Vec2 model.

        Returns:
            int: The hidden size of the model.

        """

        return self.model.config.hidden_size

    def process_signal(self, signal, sampling_rate):
        """
        Processes an audio signal and returns the extracted features.

        Args:
            signal (ndarray): The audio signal.
            sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            ndarray: The extracted features.

        """

        # inputs = self.processor(
        #     signal, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        # )
        # inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            # Inspired by https://github.com/felixbur/nkululeko/blob/main/nkululeko/feat_extract/feats_wav2vec2.py
            # Processor to normalize the input
            y = self.processor(signal, sampling_rate=sampling_rate)
            y = y["input_values"][0]
            # Use the model to extract the embeddings
            y = torch.from_numpy(y.reshape(1, -1)).to(self.device)
            # Extract the embeddings: the first element of the output are the hidden states
            y = self.model(y)[0]
            # Average the embeddings
            y = torch.mean(y, dim=1)
            # Convert the embeddings to numpy
            y = y.detach().cpu().numpy()
            # features = self.model(**inputs)
        return y.ravel()  # reshaped_features

    def free_gpu_memory(self):
        """
        Frees the GPU memory by deleting the model and processor instances, and emptying the cache.
        """
        del self.model
        del self.processor
        self.model_initialized = False


def extract_features(idx_files_augmented, feature_set, extract_options, path_features):
    """
    Extracts features from audio files based on the specified feature set.

    Args:
        idx_files_augmented (pandas.MultiIndex): audformat MultiIndex of the files to extract features from.
        feature_set (str): Name of the feature set to use for extraction.
        extract_options (dict): Additional options for feature extraction.
        path_features (str): Path to store the extracted features.

    Returns:
        pandas.DataFrame: DataFrame containing the extracted features.

    Raises:
        ValueError: If the specified feature set is not supported.
    """
    # TODO: num_workers
    extractor = None
    if feature_set in list(opensmile.FeatureSet.__members__.keys()):
        # Define the feature extractor
        extractor = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Initialize and execute audinterface
        interface = audinterface.Feature(
            feature_names=extractor.column_names,
            name=extractor.feature_set.name,
            process_func=extractor.process_signal,
            num_workers=15,  # extract_options.get("num_workers", None),
            # For progress bar
            verbose=True,
        )
    elif "wav2vec2" in feature_set:
        # Instantiate Wav2vec2
        wav2vec_feature_extractor = Wav2vec2(extract_options)
        # Determine the number of columns in the wav2vec2 embeddings
        num_columns = wav2vec_feature_extractor.get_hidden_size()
        # Generate column names in ascending order - for each embedding dimension
        column_names = [f"feature_{i}" for i in range(num_columns)]

        # Define the feature extractor for wav2vec
        extractor = {
            "process_signal": wav2vec_feature_extractor.process_signal,
            "column_names": column_names,
        }
        interface = audinterface.Feature(
            feature_names=extractor["column_names"],
            # process_func_applies_sliding_window=True,
            name="wav2vec2",
            process_func=extractor["process_signal"],
            num_workers=15,  # Or extract_options.get("num_workers", None)
            # For progress bar
            verbose=True,
        )
    else:
        raise ValueError(f"Feature set {feature_set} not supported.")
        # Could also directly set interface = opensmile.Smile() instead of extractor = opensmile.Smile()
        # TODO: for debugging: can check and compare output of extractor.process_signal() and of interface.process_index()

    # Create cache directory with feature_set as subdirectory to discern different feature sets extracted from the same index
    # Check if extract_options is None (e.g., eGeMAPS)
    if extract_options is None:
        path_cache_features = audeer.mkdir(path_features / feature_set)
    else:
        # Remove the key "device" and the path suffix if it exists
        # (better clarity when reading the model parameters from path later)
        extract_options.pop("device", None)
        extract_options.pop("path_suffix", None)

        # Format extract_options_copy to a string
        extract_options_str = "-".join(
            f"{key}_{value}" for key, value in extract_options.items()
        )

        # Compose the path
        path_cache_features = audeer.mkdir(
            path_features / feature_set / extract_options_str
        )

    # Extract features using audinterface
    df_features = interface.process_index(
        idx_files_augmented,
        preserve_index=True,
        cache_root=path_cache_features,
    )

    # Free GPU memory if a DL-based embedding is to be used
    if "wav2vec2" in feature_set:
        wav2vec_feature_extractor.free_gpu_memory()
        del wav2vec_feature_extractor
        # Call garbage collector
        gc.collect()
        # Empty the CUDA cache
        torch.cuda.empty_cache()
        # TODO: might not clean up everything

    # Filter the extracted features for rows with NaN values
    df_features = df_features.dropna(axis=0, how="all")
    # Filter the extracted features for rows with only 0 values
    df_features = df_features.loc[(df_features != 0).any(axis=1)]
    # df_features = df_features.dropna(axis=1, how="all")

    # For openSMILE: Correct column names of feature tables
    col_map = {}
    for c in sorted(df_features.columns):
        col_map[c] = re.sub("[\-\.\[\]]", "", c)
    df_features.rename(columns=col_map, inplace=True)

    return df_features
