import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_snr_distribution(df_snr, mic_prefix, out_dir, filename_prefix):
    """
    Plot SNR distributions for microphones starting with mic_prefix ("M" or "V"),
    comparing 'raw' and 'denoised' noise_status.

    Parameters
    ----------
    df_snr : pandas.DataFrame
        DataFrame containing SNR data with columns 'microphone', 'noise_status', and 'snr'.
    mic_prefix : str
        Microphone prefix to filter by ("M" for bridge microphones, "V" for radio).
    out_dir : str
        Output directory for saving plots.
    filename_prefix : str
        Prefix for output filenames.
    """
    import matplotlib as mpl

    # Set the font to a serif one
    mpl.rcParams["font.family"] = "serif"
    # Ensure fonts are embedded in the output
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 8

    df = df_snr[df_snr["microphone"].str.startswith(mic_prefix)]
    df_raw = df[df["noise_status"].isna() | (df["noise_status"] == "raw")]
    df_denoised = df[df["noise_status"] == "denoised"]

    # Print the mean and variance
    print(f"Mean and variance for {mic_prefix} microphones:")
    print("Raw microphone:")
    print(df_raw["snr"].mean())
    print(df_raw["snr"].std())
    print("Denoised microphones:")
    print(df_denoised["snr"].mean())
    print(df_denoised["snr"].std())

    # Determine common SNR range
    snr_min = min(df_raw["snr"].min(), df_denoised["snr"].min())
    snr_max = max(df_raw["snr"].max(), df_denoised["snr"].max())
    common_range = (snr_min, snr_max)
    num_bins = 30
    fixed_range = (-16, 30.01)

    plt.figure(figsize=(4.6, 2.6))
    sns.histplot(
        df_raw["snr"], kde=True, label="Raw", bins=num_bins, binrange=fixed_range
    )
    sns.histplot(
        df_denoised["snr"],
        kde=True,
        label="Denoised",
        bins=num_bins,
        binrange=fixed_range,
    )
    mic_specifier = "Bridge Microphones" if mic_prefix == "M" else "Radio Communication"
    plt.title(f"SNR Distribution - {mic_specifier}")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Number of Speech Samples")
    plt.legend()
    plt.xlim(fixed_range)  # Set x-axis limits

    os.makedirs(out_dir, exist_ok=True)
    for ext in [".png", ".pdf", ".svg", ".eps"]:
        plt.savefig(
            os.path.join(out_dir, f"{filename_prefix}{ext}"),
            bbox_inches="tight",
        )
    plt.show()
    plt.close()


def plot_all_snr_distributions(df_snr, out_dir="figures"):
    """
    Plot SNR distributions for bridge microphones ("M") and radio microphones ("V").

    Parameters
    ----------
    df_snr : pandas.DataFrame
        DataFrame containing SNR data for all microphones.
    out_dir : str, optional
        Output directory for saving plots (default is "figures").
    """
    plot_snr_distribution(
        df_snr, mic_prefix="M", out_dir=out_dir, filename_prefix="hist-snr_bridge"
    )
    plot_snr_distribution(
        df_snr, mic_prefix="V", out_dir=out_dir, filename_prefix="hist-snr_radio"
    )
