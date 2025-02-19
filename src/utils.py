def compose_experiment_str(configs):
    # Initialize an empty list to store the parts of the path
    path_parts = []

    # Iterate over the list of configuration dictionaries
    for config in configs:
        # Process the configuration dictionary and add it to the path
        path_parts.append(config_to_str(config))

    # Join the parts of the path with slashes to create the relative path
    relative_path = "/".join(path_parts)

    # Return the relative path
    return relative_path


def config_to_str(config):
    # Initialize an empty list to store the parts of the configuration string
    config_parts = []

    # Check if config is a dictionary
    if isinstance(config, dict):
        # Iterate over the configuration dictionary
        for key, value in config.items():
            # Check if the value is another dictionary
            if isinstance(value, dict):
                # Process the sub-dictionary
                sub_config = config_to_str(value)
                if sub_config != "None":
                    config_parts.append(f"{key}-{sub_config}")
                else:
                    config_parts.append(key)
            else:
                # Add the key-value pair to the configuration string, if value is not None
                if value is not None:
                    config_parts.append(f"{key}-{value}")
                else:
                    config_parts.append(key)
    else:
        # If config is not a dictionary, just return it as a string
        return str(config)

    # Join the parts of the configuration string with dashes
    config_str = "-".join(config_parts)

    # Check for list symbols and strip/replace specified characters
    if "[" in config_str and "]" in config_str:
        config_str = config_str.replace("[", "").replace("]", "")
        config_str = config_str.replace("'", "").replace('"', "")
        config_str = config_str.replace(", ", "_").replace(",", "_").replace(" ", "")

    # Return the configuration string
    return config_str


def map_modelling_config(config_modelling):
    """
    Maps the parameters tuple back to the parameter names for the modelling configuration.

    Args:
        config_modelling (tuple): A tuple containing the configuration parameters of the current modelling run.

    Returns:
        dict: A dictionary mapping the parameter names to their corresponding values.
    """
    dct_modelling_run = {
        "ModelTrainer": {
            "cohort": config_modelling[0],
            "speech_task": config_modelling[1],
            "feature_selection": config_modelling[2],
            "feature_normalization": config_modelling[3],
            "target_variable": config_modelling[4],
            "cv_config_outer": config_modelling[5],
            "cv_config_inner": config_modelling[6],
            "cv_method_inner": config_modelling[7],
            "estimator_config": config_modelling[8],
        }
    }

    return dct_modelling_run


def print_line_by_line(string):
    """
    Prints each line of the given string.

    Args:
        string (str): The input string to be printed line by line.
    """
    lines = string.split("\n")
    for line in lines:
        print(line)
