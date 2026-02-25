import os
import glob
import re
import warnings
import pandas as pd
import matplotlib as mpl

# Embed fonts in the figures
mpl.rcParams["ps.useafm"] = False
mpl.rcParams["pdf.use14corefonts"] = False
mpl.rcParams["ps.fonttype"] = 42  # Use Type 42 (TrueType) fonts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import yaml

import audeer


def extract_datetime_and_microphone(filename):
    """
    Extract date, time, and microphone from a filename like '230205,015956,M1,9510682_0.wav'.
    Returns a tuple: (datetime64, microphone) or (pd.NaT, None) if not matched.
    """
    match = re.match(r"(?P<date>\d{6}),(?P<time>\d{6}),(?P<microphone>[^,]+)", filename)
    if match:
        date_str = match.group("date")
        time_str = match.group("time")
        mic = match.group("microphone")
        # Combine date and time into a single datetime object
        dt = pd.to_datetime(date_str + time_str, format="%y%m%d%H%M%S", errors="coerce")
        return dt, mic
    else:
        warnings.warn(f"Filename '{filename}' cannot be converted to DateTime.")
        return pd.NaT, None


def concat_all_preds(dir_output_root, dir_evaluated, flag_concat, preds_str="emotion"):
    """
    Concatenates individual prediction DataFrames from multiple subdirectories into a single DataFrame.

    This function searches for prediction directories within the specified output root directory,
    loads prediction dictionaries from pickle files, and concatenates all DataFrames into one.
    It also extracts datetime and microphone information from the file index and adds them as columns.
    The resulting DataFrame is saved as a pickle file in the evaluated directory.

    Parameters
    ----------
    dir_output_root : str
        Path to the root directory containing subdirectories with prediction results.
    dir_evaluated : str
        Path to the directory where the concatenated DataFrame will be saved or loaded from.
    flag_concat : str
        If "false", loads an existing concatenated DataFrame from disk instead of recomputing.
        Otherwise, performs concatenation.
    preds_str : str, optional
        String specifying the type of predictions to look for (default is "emotion").

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame containing all predictions, with additional 'time' and 'microphone' columns.

    Notes
    -----
    - If the concatenated DataFrame already exists and `flag_concat` is "false", it is loaded from disk.
    - If a prediction file does not exist in a subdirectory, a warning is issued.
    - The 'time' column is currently set to the extracted base datetime for each file.
    """

    # The output path for the concatenated DataFrame
    output_path = os.path.join(dir_evaluated, f"df_all_predictions_{preds_str}.pkl")
    if flag_concat == "false":
        print(f"Loading existing DataFrame from {output_path}.")
        df_all_preds = pd.read_pickle(output_path)
        return df_all_preds

    # Iterate through all subdirectories in the output root directory
    # and look for the specified predictions directory.
    df_all_preds = pd.DataFrame()
    for dir_preds in glob.glob(
        os.path.join(dir_output_root, "*", f"predictions-{preds_str}")
    ):
        file_dct_preds = (
            f"dct_predictions_{os.path.basename(os.path.dirname(dir_preds))}.pkl"
        )
        path_dct_preds = os.path.join(dir_preds, file_dct_preds)

        if os.path.exists(path_dct_preds):
            dct_predictions = pd.read_pickle(path_dct_preds)
            # Check if keys contain noise status
            new_dfs = []
            for mic_key, df in dct_predictions.items():
                if "_" in mic_key:
                    mic_name, noise_status = mic_key.split("_", 1)
                else:
                    mic_name = mic_key
                    noise_status = None
                df = df.copy()
                df["noise_status"] = noise_status
                new_dfs.append(df)
            df_all_preds = pd.concat([df_all_preds, *new_dfs])
        else:
            warnings.warn(f"File {path_dct_preds} does not exist.")

    # Extract datetime and microphone using the helper function
    extracted = df_all_preds.index.get_level_values("file").map(
        extract_datetime_and_microphone
    )
    base_times = pd.Series([dt for dt, mic in extracted], index=df_all_preds.index)
    microphones = [mic for dt, mic in extracted]

    # # Add the start offset to the extracted base time
    # df_all_preds["time"] = base_times + start_times

    df_all_preds["time"] = base_times
    df_all_preds["microphone"] = microphones

    # Save the concatenated DataFrame to a pickle file
    df_all_preds.to_pickle(output_path)

    return df_all_preds


def load_database_aisl(path_active):
    """
    Loads a CSV file containing AI SoundLab metadata into a pandas DataFrame.

    Args:
        path_active (str): The file path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the specified CSV file.
    """
    df_files = pd.read_csv(path_active)

    # Set the index column
    index_column = "file"
    df_files = df_files.set_index(index_column)

    # Convert date-related columns into datetime format
    if "date_file" in df_files.columns:
        df_files["date_file"] = pd.to_datetime(df_files["date_file"])
    if "date_survey" in df_files.columns:
        df_files["date_survey"] = pd.to_datetime(df_files["date_survey"])

    # Filter out participants that have less than x sessions
    print(
        f"Number of participants before filtering out too few sessions: {len(df_files['participant_code'].unique())}; "
        f"Shape of the DataFrame before filtering: {df_files.shape}"
    )
    min_sessions = 1
    unique_sessions_per_participant = df_files.groupby("participant_code")[
        "session"
    ].nunique()
    participants_too_few_sessions = unique_sessions_per_participant[
        unique_sessions_per_participant <= min_sessions
    ]
    list(participants_too_few_sessions.index)

    # Filter the main DataFrame
    df_files = df_files[
        ~df_files["participant_code"].isin(list(participants_too_few_sessions.index))
    ]
    print(
        f"Number of participants after filtering out too few sessions: {len(df_files['participant_code'].unique())}; "
        f"Shape of the DataFrame after filtering: {df_files.shape}"
    )

    # Filter out particular prompts
    discard_prompts = [
        "sustained_utterance_0113",
        "emotion_acting_3130",
        "emotion_acting_5177",
        "emotion_acting_2017",
    ]
    df_files = df_files[~df_files.prompt.isin(discard_prompts)]
    print(f"Shape of the DataFrame after filtering prompts: {df_files.shape}")

    return df_files


def prepare_event_markers(event_timings, freq="D"):
    """
    Prepare a list of event marker dicts for plotting.

    Parameters
    ----------
    event_timings : dict
        Dictionary containing event timing information with 'land_and_sea' and 'loading_and_discharge' keys.
    freq : str, optional
        Frequency string for time floor operation (default is "D" for daily).

    Returns
    -------
    list
        List of dictionaries, each containing: {'date': pd.Timestamp, 'color': str, 'label': str,
        'linestyle': str, 'alpha': float}. Used for plotting event markers on time series plots.
    """
    markers = []
    # Land and sea (grey, label "L")
    for entry in event_timings.get("land_and_sea", {}).get("land", []):
        start = pd.to_datetime(entry["start"]).floor(freq)
        markers.append(
            {
                "date": start,
                "color": "grey",
                "label": "L",
                "linestyle": "dashed",
                "alpha": 0.5,
            }
        )
        end = pd.to_datetime(entry["end"]).floor(freq)
        markers.append(
            {
                "date": end,
                "color": "grey",
                "label": None,
                "linestyle": "dashed",
                "alpha": 0.5,
            }
        )
    # Loading (red, label "O")
    for entry in event_timings.get("loading_and_discharge", {}).get("loading", []):
        start = pd.to_datetime(entry["start"]).floor(freq)
        markers.append(
            {
                "date": start,
                "color": "red",
                "label": "O",
                "linestyle": "dashed",
                "alpha": 0.5,
            }
        )
        end = pd.to_datetime(entry["end"]).floor(freq)
        markers.append(
            {
                "date": end,
                "color": "red",
                "label": None,
                "linestyle": "dashed",
                "alpha": 0.5,
            }
        )
    # Discharge (red, label "O")
    for entry in event_timings.get("loading_and_discharge", {}).get("discharge", []):
        start = pd.to_datetime(entry["start"]).floor(freq)
        markers.append(
            {
                "date": start,
                "color": "red",
                "label": "O",
                "linestyle": "dashed",
                "alpha": 0.5,
            }
        )
        end = pd.to_datetime(entry["end"]).floor(freq)
        markers.append(
            {
                "date": end,
                "color": "red",
                "label": None,
                "linestyle": "dashed",
                "alpha": 0.5,
            }
        )
    return markers


def plot_event_markers(
    ax, markers, x_dates, draw_hspans=True, hspan_ypos=0.90, force_y01=True
):
    """
    Plot vertical lines and labels for events on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis to plot on.
    markers : list
        List of marker dictionaries from prepare_event_markers().
    x_dates : list
        List of date strings (x-axis labels) as used in the plot.
    draw_hspans : bool, optional
        If True, draw horizontal lines for event periods (default is True).
    hspan_ypos : float, optional
        Relative y position for horizontal lines and label (e.g., 0.90 = 90% of y-axis, default is 0.90).
    force_y01 : bool, optional
        If True, force y-axis to [0, 1] (default is True).
    """
    # Optionally force y-axis to [0, 1] before plotting markers
    if force_y01:
        ax.set_ylim(0, 1)

    # Get y-limits and compute y for lines/labels
    ylim = ax.get_ylim()
    y_marker = ylim[0] + (ylim[1] - ylim[0]) * hspan_ypos
    y_label = y_marker - 0.05 * (ylim[1] - ylim[0])  # 5% below the horizontal line

    # Group markers by event (start/end pairs)
    paired_events = []
    temp = {}
    for marker in markers:
        date_str = marker["date"].strftime("%Y-%m-%d")
        if marker["label"]:  # start
            temp = marker.copy()
            temp["start_str"] = date_str
        else:  # end
            if temp:
                temp["end_str"] = date_str
                temp["end_color"] = marker["color"]
                paired_events.append(temp)
                temp = {}

    # Draw vertical lines and labels
    label_offsets = {}  # Track number of labels per xpos
    for marker in markers:
        date_str = marker["date"].strftime("%Y-%m-%d")
        if date_str in x_dates:
            xpos = x_dates.index(date_str)
        else:
            if len(x_dates) == 0:
                continue
            if marker["date"] < pd.to_datetime(x_dates[0]):
                xpos = 0
            elif marker["date"] > pd.to_datetime(x_dates[-1]):
                xpos = len(x_dates) - 1
            else:
                date_diffs = [
                    abs((pd.to_datetime(date_str) - pd.to_datetime(xd)).days)
                    for xd in x_dates
                ]
                xpos = int(np.argmin(date_diffs))
        # Draw vertical line from bottom to top of y-axis
        ax.vlines(
            xpos,
            ylim[0],
            ylim[1],
            color=marker["color"],
            linestyle=marker["linestyle"],
            alpha=marker["alpha"],
            linewidth=1.5,
        )
        if marker["label"]:
            # Offset label horizontally if multiple labels at same xpos
            offset = label_offsets.get(xpos, 0)
            ax.text(
                xpos + 0.05 + 0.10 * offset,
                y_label,
                marker["label"],
                color=marker["color"],
                rotation=90,
                va="top",
                ha="left",
                fontsize=9,
                alpha=0.7,
                clip_on=True,
            )
            label_offsets[xpos] = offset + 1

    # Draw horizontal spans
    if draw_hspans:
        for event in paired_events:
            # Determine x_start
            if event["start_str"] in x_dates:
                x_start = x_dates.index(event["start_str"])
            else:
                if len(x_dates) == 0:
                    continue
                if pd.to_datetime(event["start_str"]) < pd.to_datetime(x_dates[0]):
                    x_start = -0.5
                elif pd.to_datetime(event["start_str"]) > pd.to_datetime(x_dates[-1]):
                    x_start = len(x_dates) - 0.5
                else:
                    date_diffs = [
                        abs(
                            (
                                pd.to_datetime(event["start_str"]) - pd.to_datetime(xd)
                            ).days
                        )
                        for xd in x_dates
                    ]
                    x_start = int(np.argmin(date_diffs))
            # Determine x_end
            if event["end_str"] in x_dates:
                x_end = x_dates.index(event["end_str"])
            else:
                if len(x_dates) == 0:
                    continue
                if pd.to_datetime(event["end_str"]) < pd.to_datetime(x_dates[0]):
                    x_end = -0.5
                elif pd.to_datetime(event["end_str"]) > pd.to_datetime(x_dates[-1]):
                    x_end = len(x_dates) - 0.5
                else:
                    date_diffs = [
                        abs(
                            (pd.to_datetime(event["end_str"]) - pd.to_datetime(xd)).days
                        )
                        for xd in x_dates
                    ]
                    x_end = int(np.argmin(date_diffs))
            ax.hlines(
                y_marker,
                x_start,
                x_end,
                color=event["color"],
                linestyle=event["linestyle"],
                alpha=event["alpha"],
                linewidth=3,
            )


def normalize_column(series, min_raw, max_raw):
    """
    Normalize a pandas Series to [0, 1] given its raw min and max.

    Parameters
    ----------
    series : pandas.Series
        The data series to normalize.
    min_raw : float
        The minimum value of the raw data range.
    max_raw : float
        The maximum value of the raw data range.

    Returns
    -------
    pandas.Series
        Normalized series with values scaled to [0, 1] range.
    """
    return (series - min_raw) / (max_raw - min_raw)


def plot_active_time_course_boxplots(
    df_session,
    date_col,
    value_cols,
    value_ranges,
    freq="D",
    title_prefix="",
    out_dir=None,
    event_markers=None,
):
    """
    Create time series box plots for actively collected survey data.

    Parameters
    ----------
    df_session : pandas.DataFrame
        DataFrame containing session-wise survey data.
    date_col : str
        Name of the column containing date information.
    value_cols : list
        List of column names to plot.
    value_ranges : dict
        Dictionary mapping column names to (min, max) tuples for normalization.
    freq : str, optional
        Frequency for time binning (default is "D" for daily).
    title_prefix : str, optional
        Prefix for plot titles (default is "").
    out_dir : str, optional
        Output directory for saving plots (default is None).
    event_markers : list, optional
        List of event markers for plotting (default is None).
    """
    import matplotlib as mpl

    # Set the font to a serif one
    mpl.rcParams["font.family"] = "serif"
    # Ensure fonts are embedded in the output
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 14

    df_session = df_session.copy()
    df_session = df_session[df_session[date_col].notna()]
    df_session["date_bin"] = pd.to_datetime(df_session[date_col]).dt.floor(freq)
    # Normalize columns
    for col, (min_raw, max_raw) in value_ranges.items():
        norm_col = f"{col}_norm"
        df_session[norm_col] = normalize_column(df_session[col], min_raw, max_raw)
    # Plot for each value column (normalized)
    for col in value_cols:
        norm_col = f"{col}_norm"
        plt.figure(figsize=(15, 5))
        df_session = df_session.sort_values("date_bin")
        df_session["date_bin_str"] = df_session["date_bin"].dt.strftime("%Y-%m-%d")

        col_label = col.replace("_", " ").title()

        # Count samples per bin
        counts = df_session.groupby("date_bin_str")[norm_col].count()
        x_labels = [
            f"{d} (n={counts[d]:,})" for d in counts.index
        ]  # thousands separator

        ax = sns.boxplot(
            data=df_session,
            x="date_bin_str",
            y=norm_col,
            color="lightgreen",
        )
        plt.xticks(
            ticks=np.arange(len(x_labels)),
            labels=x_labels,
            rotation=90,
            fontsize=10,
        )
        # Plot event markers if provided
        if event_markers:
            x_dates = list(df_session["date_bin_str"].unique())
            plot_event_markers(ax, event_markers, x_dates)
        plt.title(f"{title_prefix}{col_label} (normalized) by Day", fontsize=28)
        plt.xlabel("Date")
        plt.ylabel(f"{col_label} (normalized)")
        plt.tight_layout()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path_composed = os.path.join(out_dir, f"{title_prefix}{col}_by_day")
            plt.savefig(f"{path_composed}.png")
            plt.savefig(f"{path_composed}.eps")
            plt.savefig(f"{path_composed}.pdf")
        plt.show()
        plt.close()


def plot_passive_time_course_boxplots(
    df,
    time_col,
    value_cols,
    group_col,
    groupings,
    freq="D",
    title_prefix="",
    out_dir=None,
    event_markers=None,
):
    """
    Create time series box plots for passively collected audio data predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing passive prediction data.
    time_col : str
        Name of the column containing time information.
    value_cols : list
        List of column names to plot (e.g., emotion predictions).
    group_col : str
        Name of the column used for grouping (e.g., microphone channel).
    groupings : dict
        Dictionary mapping group labels to selection criteria.
    freq : str, optional
        Frequency for time binning (default is "D" for daily).
    title_prefix : str, optional
        Prefix for plot titles (default is "").
    out_dir : str, optional
        Output directory for saving plots (default is None).
    event_markers : list, optional
        List of event markers for plotting (default is None).
    """
    # Set the font to a serif one
    mpl.rcParams["font.family"] = "serif"
    # Ensure fonts are embedded in the output
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 18

    df = df.copy()
    df = df[df[time_col].notna()]
    df["date_bin"] = df[time_col].dt.floor(freq)
    df["date_bin"] = pd.to_datetime(df["date_bin"])

    for group_label, group_selector in groupings.items():
        if group_selector is None:
            df_group = df
        elif callable(group_selector):
            df_group = df[group_selector(df[group_col])]
        else:
            df_group = df[df[group_col].isin(group_selector)]

        if df_group.empty:
            continue

        for value_col in value_cols:
            plt.figure(figsize=(15, 5))
            df_group = df_group.copy()
            if freq == "D":
                df_group["date_bin_str"] = df_group["date_bin"].dt.strftime("%Y-%m-%d")
                x_col = "date_bin_str"
            else:
                x_col = "date_bin"
            # Sort by x_col to ensure correct order on x-axis
            df_group = df_group.sort_values(x_col)

            # Count samples per bin
            counts = df_group.groupby(x_col)[value_col].count()
            x_labels = [
                f"{d} (n={counts[d]:,})" for d in counts.index
            ]  # thousands separator

            # Compose title descriptor
            value_col_label = value_col.replace("_", " ").title()

            ax = sns.boxplot(
                data=df_group,
                x=x_col,
                y=value_col,
                color="skyblue",
            )
            plt.xticks(
                ticks=np.arange(len(x_labels)),
                labels=x_labels,
                rotation=90,
                fontsize=10,
            )
            # Plot event markers if provided
            if event_markers:
                x_dates = list(df_group[x_col].unique())
                plot_event_markers(ax, event_markers, x_dates)
            plt.title(f"{title_prefix}{value_col_label} by Day", fontsize=28)
            plt.xlabel("Date")
            plt.ylabel(value_col_label)
            plt.tight_layout()
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                path_base = os.path.join(
                    out_dir, f"{title_prefix}{value_col}_by_day_{group_label}"
                )
                plt.savefig(f"{path_base}.png")
                plt.savefig(f"{path_base}.eps")
                plt.savefig(f"{path_base}.pdf")
            plt.show()
            plt.close()


def plot_boxplots(
    df_passive, df_active, event_markers, event_timings, dir_evaluated, preds_str
):
    """
    Orchestrate the creation of all time series and aggregated box plots.

    Parameters
    ----------
    df_passive : pandas.DataFrame
        DataFrame containing passive prediction data.
    df_active : pandas.DataFrame
        DataFrame containing active survey data.
    event_markers : list
        List of event markers for plotting.
    event_timings : dict
        Dictionary containing event timing information.
    dir_evaluated : str
        Directory path for saving evaluation results.
    preds_str : str
        String identifier for prediction type (e.g., "emotion").
    """

    # Passive
    value_cols_passive = [
        "prediction_arousal",
        "prediction_dominance",
        "prediction_valence",
    ]
    groupings = {
        "all": None,
        "bridge_M": lambda s: s.str.startswith("M"),
        "radio_V": lambda s: s.str.startswith("V"),
    }
    dir_out_passive = os.path.join(
        dir_evaluated, f"plots-time_course-passive-{preds_str}"
    )
    dir_out_passive = audeer.mkdir(dir_out_passive)
    plot_passive_time_course_boxplots(
        df=df_passive,
        time_col="time",
        value_cols=value_cols_passive,
        group_col="microphone",
        groupings=groupings,
        freq="D",
        title_prefix="Passive - ",
        out_dir=dir_out_passive,
        event_markers=event_markers,
    )

    # Align active data to passive time range and aggregate session-wise
    passive_min = df_passive["time"].min()
    passive_max = df_passive["time"].max()
    df_active_aligned = df_active[
        (df_active["date_survey"] >= passive_min)
        & (df_active["date_survey"] <= passive_max)
    ].copy()
    if "session" in df_active_aligned.columns:
        df_active_session = (
            df_active_aligned.sort_values("date_survey")
            .groupby("session")
            .first()
            .reset_index()
        )
    else:
        df_active_session = df_active_aligned.copy()

    # Active (time course, session-wise)
    value_cols_active = [
        "phq_8_total_score",
        "stress_current",
        "stress_work_tasks",
        "pss_10_total_score",
        "who_5_percentage_score_corrected",
        "emotions_current_happy",
        "emotions_current_sad",
        "emotions_current_anger",
        "emotions_current_fear",
        "emotions_current_indifference",
        "emotions_current_shame",
    ]
    value_ranges = {
        "phq_8_total_score": (0, 24),
        "stress_current": (0, 100),
        "stress_work_tasks": (0, 100),
        "pss_10_total_score": (0, 40),
        "who_5_percentage_score_corrected": (0, 100),
        "emotions_current_happy": (0, 100),
        "emotions_current_sad": (0, 100),
        "emotions_current_anger": (0, 100),
        "emotions_current_fear": (0, 100),
        "emotions_current_indifference": (0, 100),
        "emotions_current_shame": (0, 100),
    }
    dir_out_active = os.path.join(dir_evaluated, f"plots-time_course-active")
    dir_out_active = audeer.mkdir(dir_out_active)
    plot_active_time_course_boxplots(
        df_session=df_active_session,
        date_col="date_survey",
        value_cols=value_cols_active,
        value_ranges=value_ranges,
        freq="D",
        title_prefix="Active - ",
        out_dir=dir_out_active,
        event_markers=event_markers,
    )

    # Aggregated event boxplots (session-wise, time-window-filtered)
    dir_out_agg = os.path.join(dir_evaluated, "plots-aggregated-events-passive")
    plot_aggregated_event_boxplots(
        df_passive,
        event_timings,
        out_dir=dir_out_agg,
        title_prefix="Aggregated - ",
    )

    dir_out_agg_active = os.path.join(dir_evaluated, "plots-aggregated-events-active")
    plot_aggregated_event_boxplots_active(
        df_active_session,  # session-wise, time-window-filtered
        event_timings,
        value_ranges=value_ranges,
        out_dir=dir_out_agg_active,
        title_prefix="Aggregated - Active - ",
    )

    return


def subtract_intervals(base_intervals, subtract_intervals):
    """
    Subtract a list of intervals from another list of intervals.

    Parameters
    ----------
    base_intervals : list
        List of tuples (start, end) representing base time intervals.
    subtract_intervals : list
        List of tuples (start, end) representing intervals to subtract.

    Returns
    -------
    list
        List of intervals in base_intervals that do not overlap with any in subtract_intervals.
        Each interval is a tuple (start, end).
    """
    result = []
    for base_start, base_end in base_intervals:
        current = [(base_start, base_end)]
        for sub_start, sub_end in subtract_intervals:
            new_current = []
            for seg_start, seg_end in current:
                # No overlap
                if seg_end <= sub_start or seg_start >= sub_end:
                    new_current.append((seg_start, seg_end))
                else:
                    # Overlap: split segment if needed
                    if seg_start < sub_start:
                        new_current.append((seg_start, sub_start))
                    if seg_end > sub_end:
                        new_current.append((sub_end, seg_end))
            current = new_current
        result.extend(current)
    return result


def plot_aggregated_event_boxplots(
    df_passive,
    event_timings,
    value_cols=None,
    out_dir=None,
    title_prefix="Aggregated - ",
):
    """
    Plot boxplots for all data points during 'land', 'sea', and 'operation' (loading/discharge) periods,
    for all microphones, bridge microphones (M), and radio microphones (V).
    """
    if value_cols is None:
        value_cols = [
            "prediction_arousal",
            "prediction_dominance",
            "prediction_valence",
        ]

    def collect_intervals(event_list):
        return [
            (pd.to_datetime(e["start"]), pd.to_datetime(e["end"])) for e in event_list
        ]

    land_intervals_raw = collect_intervals(
        event_timings.get("land_and_sea", {}).get("land", [])
    )
    sea_intervals = collect_intervals(
        event_timings.get("land_and_sea", {}).get("sea", [])
    )
    loading_intervals = collect_intervals(
        event_timings.get("loading_and_discharge", {}).get("loading", [])
    )
    discharge_intervals = collect_intervals(
        event_timings.get("loading_and_discharge", {}).get("discharge", [])
    )
    operation_intervals = loading_intervals + discharge_intervals
    # Exclude loading operation from being on land
    land_intervals = subtract_intervals(land_intervals_raw, operation_intervals)

    def filter_by_intervals(df, intervals, time_col="time"):
        mask = pd.Series(False, index=df.index)
        for start, end in intervals:
            mask |= (df[time_col] >= start) & (df[time_col] <= end)
        return df[mask]

    groupings = {
        "all": None,
        "bridge_M": lambda s: s.str.startswith("M"),
        "radio_V": lambda s: s.str.startswith("V"),
    }

    for group_label, group_selector in groupings.items():
        if group_selector is None:
            df_group = df_passive
        else:
            df_group = df_passive[group_selector(df_passive["microphone"])]

        if df_group.empty:
            continue

        df_land = filter_by_intervals(df_group, land_intervals)
        df_sea = filter_by_intervals(df_group, sea_intervals)
        df_operation = filter_by_intervals(df_group, operation_intervals)

        for col in value_cols:
            # Count samples per category
            counts = {
                "Land": df_land[col].dropna().shape[0],
                "Sea": df_sea[col].dropna().shape[0],
                "Operation": df_operation[col].dropna().shape[0],
            }
            # Prepare data for plotting
            data = []
            for label, df_cat in zip(
                ["Land", "Sea", "Operation"], [df_land, df_sea, df_operation]
            ):
                for val in df_cat[col]:
                    data.append({"Category": label, "Value": val})
            df_plot = pd.DataFrame(data)

            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(
                data=df_plot,
                x="Category",
                y="Value",
                palette="Set2",
            )
            # Set x-tick labels with formatted counts
            x_labels = [
                f"{cat} (n={counts[cat]:,})" for cat in ["Land", "Sea", "Operation"]
            ]
            ax.set_xticklabels(x_labels)
            plt.title(f"{title_prefix}{col} ({group_label})")
            plt.xlabel("Event Category")
            plt.ylabel(col)
            plt.tight_layout()
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                path_composed = os.path.join(
                    out_dir, f"{title_prefix}{col}_aggregated_{group_label}"
                )
                plt.savefig(f"{path_composed}.png")
                plt.savefig(f"{path_composed}.eps")
                plt.savefig(f"{path_composed}.pdf")
            plt.show()
            plt.close()


def plot_aggregated_event_boxplots_active(
    df_active,
    event_timings,
    value_cols=None,
    value_ranges=None,
    out_dir=None,
    title_prefix="Aggregated - Active - ",
):
    """
    Plot boxplots for all data points during 'land', 'sea', and 'operation' (loading/discharge) periods
    for the active (survey) data, with y-axis limits set by value_ranges if provided.
    """
    if value_cols is None:
        value_cols = [
            "phq_8_total_score",
            "stress_current",
            "stress_work_tasks",
            "pss_10_total_score",
            "who_5_percentage_score_corrected",
            "emotions_current_happy",
            "emotions_current_sad",
            "emotions_current_anger",
            "emotions_current_fear",
            "emotions_current_indifference",
            "emotions_current_shame",
        ]

    def collect_intervals(event_list):
        return [
            (pd.to_datetime(e["start"]), pd.to_datetime(e["end"])) for e in event_list
        ]

    land_intervals_raw = collect_intervals(
        event_timings.get("land_and_sea", {}).get("land", [])
    )

    sea_intervals = collect_intervals(
        event_timings.get("land_and_sea", {}).get("sea", [])
    )
    loading_intervals = collect_intervals(
        event_timings.get("loading_and_discharge", {}).get("loading", [])
    )
    discharge_intervals = collect_intervals(
        event_timings.get("loading_and_discharge", {}).get("discharge", [])
    )
    operation_intervals = loading_intervals + discharge_intervals
    # Exclude loading operation from being on land
    land_intervals = subtract_intervals(land_intervals_raw, operation_intervals)

    def filter_by_intervals(df, intervals, time_col="date_survey"):
        mask = pd.Series(False, index=df.index)
        for start, end in intervals:
            mask |= (df[time_col] >= start) & (df[time_col] <= end)
        return df[mask]

    df_land = filter_by_intervals(df_active, land_intervals)
    df_sea = filter_by_intervals(df_active, sea_intervals)
    df_operation = filter_by_intervals(df_active, operation_intervals)

    for col in value_cols:
        # Count samples per category
        counts = {
            "Land": df_land[col].dropna().shape[0],
            "Sea": df_sea[col].dropna().shape[0],
            "Operation": df_operation[col].dropna().shape[0],
        }
        # Prepare data for plotting
        data = []
        for label, df_cat in zip(
            ["Land", "Sea", "Operation"], [df_land, df_sea, df_operation]
        ):
            for val in df_cat[col]:
                data.append({"Category": label, "Value": val})
        df_plot = pd.DataFrame(data)

        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(
            data=df_plot,
            x="Category",
            y="Value",
            palette="Set2",
        )
        # Set x-tick labels with formatted counts
        x_labels = [
            f"{cat} (n={counts[cat]:,})" for cat in ["Land", "Sea", "Operation"]
        ]
        ax.set_xticklabels(x_labels)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("Event Category")
        plt.ylabel(col)
        if value_ranges and col in value_ranges:
            plt.ylim(value_ranges[col])
        plt.tight_layout()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path_composed = os.path.join(
                out_dir, f"{title_prefix}{col}_aggregated_active"
            )
            plt.savefig(f"{path_composed}.png")
            plt.savefig(f"{path_composed}.eps")
            plt.savefig(f"{path_composed}.pdf")
        plt.show()
        plt.close()


def filter_duration(df, min_duration):
    """
    Filter the DataFrame to only include rows where the 'duration' column is greater than or equal to min_duration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with MultiIndex containing 'start' and 'end' levels for calculating duration.
    min_duration : float
        Minimum duration threshold in seconds.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only segments with duration >= min_duration.
    """
    print(f"Filtering segments with duration >= {min_duration} seconds.")
    print(f"Original number of segments: {len(df)}")
    durations = (
        df.index.get_level_values("end") - df.index.get_level_values("start")
    ).total_seconds()
    print(f"Number of segments after filtering: {len(df[durations >= min_duration])}")

    return df[durations >= min_duration]


def evaluate_time_course(
    df_passive, df_active, path_time_course_events, dir_evaluated, preds_str
):
    """
    Main function to evaluate and plot time course data for both passive and active data collection.

    Parameters
    ----------
    df_passive : pandas.DataFrame
        DataFrame containing passive prediction data from VDR audio analysis.
    df_active : pandas.DataFrame
        DataFrame containing active survey data.
    path_time_course_events : str
        Path to YAML file containing event timing information.
    dir_evaluated : str
        Directory path for saving evaluation results and plots.
    preds_str : str
        String identifier for prediction type (e.g., "emotion").
    """
    # Load the time course events from a YAML file
    with open(path_time_course_events, "r") as f:
        event_timings = yaml.safe_load(f)

    event_markers = prepare_event_markers(event_timings, freq="D")

    plot_boxplots(
        df_passive, df_active, event_markers, event_timings, dir_evaluated, preds_str
    )
