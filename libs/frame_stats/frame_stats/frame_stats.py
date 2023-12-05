import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from numpy.typing import ArrayLike
from numbers import Number
from tqdm import tqdm
from typing import Callable, Tuple, Dict


def bootstrap_ci(data: ArrayLike,
                 statistic: Callable[[ArrayLike], float],
                 n_samples: int,
                 alpha: float,
                 method: str = "residual",
                 seed: int = None) -> Dict:
    """
    Generate a bootstrapped confidence interval and point estimate for the
    statistic from the data.

    Parameters
    ----------
    data: ArrayLike[Number]
        The data from which to calculate the statistic
    statistic: Callable[[ArrayLike], float]
        The value we want to estimate from the data provided as a function that
        takes an array and returns a float
    n_samples: int
        Number of bootstrapped samples to use to estimate the confidence interval
    alpha: float
        Sognifigance. Probability of type I error
    method: str ()
        Which confidence interval approach to use
    seed: int
        Seed for random number generator


    Returns
    -------
    result: dict
        Contains three fields: "estimate", the estimate of our statistic,
        "lower", the lower bound of our confidence interval, and "upper", the
        upper bound of our confidence interval
    """

    # the value we will return
    result = {}

    # calculate the point estimate
    result["estimate"] = statistic(data)

    # now the long part
    rng = np.random.default_rng(seed)
    boots_statistic = np.zeros(n_samples)

    for s in tqdm(range(n_samples)):
        boot_sample = rng.choice(data, size=data.shape, replace=True)

        boots_statistic[s] = statistic(boot_sample)

    if method == "percentile":
        result["lower"], result["upper"] = percentile_ci(boots_statistic,
                                                         alpha)
    elif method == "residual":
        result["lower"], result["upper"] = residual_ci(boots_statistic,
                                                       alpha,
                                                       result["estimate"])
    # want to implement BCA also
    else:
        raise ValueError("method must be in  ['percentile', 'residual']")

    return result

def bootstrap_ci_multivariate(df,
                               statistic: Callable[[pd.DataFrame], float],
                               n_samples: int,
                               sample_axis: str,
                               alpha: float,
                               method: str = "residual",
                               seed: int = None) -> Dict:
    
    result = {}
    result["estimate"] = statistic(df)

    # empty arrays for bootstrapped stats
    boot_statistics = np.zeros(n_samples)
    
    # check how to sample
    rng = np.random.default_rng(seed)
    if sample_axis == "rows":
        sample_size = df.shape[0]

    elif sample_axis == "columns":
        sample_size = df.shape[1]
    
    else:
        raise ValueError("sample_axis must be in ['rows', 'columns;]")

    # run the bootstrapping
    for s in tqdm(range(n_samples)):

        boot_sample = df.sample(n=sample_size, axis=sample_axis, replace=True)
        boot_statistics[s] = statistic(boot_sample)

    
    if method == "percentile":
        result["lower"], result["upper"] = percentile_ci(boot_statistics,
                                                         alpha)
    elif method == "residual":
        result["lower"], result["upper"] = residual_ci(boot_statistics,
                                                       alpha,
                                                       result["estimate"])
    # want to implement BCA also
    else:
        raise ValueError("method must be in  ['percentile', 'residual']")

    return result


def percentile_ci(boot_estimates: ArrayLike,
                  alpha: float) -> Tuple[float, float]:

    lower = np.quantile(boot_estimates, alpha / 2)
    upper = np.quantile(boot_estimates, 1 - (alpha / 2))

    return (lower, upper)


def residual_ci(boot_estimates: ArrayLike,
                alpha: float,
                point_estimate: float) -> Tuple[float, float]:

    lower = 2 * point_estimate - np.quantile(boot_estimates, 1 - (alpha / 2))
    upper = 2 * point_estimate - np.quantile(boot_estimates, alpha / 2)

    return (lower, upper)


def draw_frame_frequencies(frame_df,
                           plot_frames,
                           ax,
                           group_col="Group",
                           frame_col="Frame",
                           estimate_col="Frequency",
                           lower_col="lower",
                           upper_col="upper",
                           legend_loc="upper right",
                           colors=["C0", "C1", "C2", "C3"]):
    
    # pull out the columns
    frame_df = frame_df[frame_df[frame_col].isin(plot_frames)]
    

    # layout features
    bar_width = 1
    pad = 2 * bar_width
    xmax = min(frame_df[upper_col].max() + 0.05, 1.0)

    # useful for calculations
    groups = frame_df[group_col].unique()
    frames = frame_df[frame_col].unique()
    print(groups)

    gen_y_coords = [np.arange(i,
                              (len(groups)+pad) * len(frames) + i,
                              len(groups)+pad)
                    for i in range(len(groups))]

    gen_yticks = np.arange(1.5,
                           (len(groups)+pad) * len(frames),
                           len(groups)+pad)

    for i, group in enumerate(groups):
        # generic frame probabilities
        group_long = frame_df[frame_df[group_col] == group]
        sorted_group = group_long.sort_values(
            by=frame_col, axis=0, ascending=False)
    
        # point estimates
        ax.barh(gen_y_coords[i], sorted_group[estimate_col],
                height=bar_width,
                color=colors[i],
                label=f"{group.title()}")
    
        # confidence intervals
        ax.hlines(gen_y_coords[i],
                  sorted_group[lower_col],
                  sorted_group[upper_col],
                  color="black")
    
        
    ax.set_yticks(gen_yticks)
    ax.set_yticklabels(sorted_group[frame_col])
    ax.set_xlabel("Frequency Cued by Tweets")
    ax.set_xlim(0, xmax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),
            title='Group', loc=legend_loc)

    return ax


def load_all_frames(paths: Dict, config: Dict) -> pd.DataFrame:
    groups = ["congress", "journalists", "trump", "public"]
    frame_types = ["generic", "specific", "narrative"]
    all_frame_dfs = []

    frames = {k: {} for k in groups}
    for group in tqdm(groups):
        type_dfs = []
        if group != "public":
            for frame_type in frame_types:

                # all frames are in one file for the public
                # we have to merge a couple of different files in either case
                if group != "public":
                    # predicted frames
                    predictions = pd.read_csv(paths[group]["frames"][frame_type],
                                              sep="\t").drop("text", axis="columns")

                    # time stamps for granger causality
                    tweet_info = pd.read_csv(paths[group]["full_tweets"],
                                             sep="\t")[["id_str", "time_stamp", "screen_name"]]

                    tweet_info["id_str"] = tweet_info["id_str"].astype(str)
                    predictions["id_str"] = predictions["id_str"].astype(str)

                    # merge and add to list of dataframes to combine
                    tweet_df = pd.merge(tweet_info, predictions,
                                        how="right", on="id_str")
                    tweet_df["group"] = group
                    type_dfs.append(tweet_df)

            all_frames_for_group = reduce(lambda l, r: pd.merge(l, r,
                                                                on=["id_str",
                                                                    "time_stamp",
                                                                    "screen_name",
                                                                    "group"]),
                                          type_dfs)

        else:
            # frame predictions
            predictions = pd.read_csv(paths["public"]["frames"]["all"],
                                      sep="\t")
            tweet_info = pd.read_csv("data/us_public_ids.tsv", sep="\t")
            tweet_info["time_stamp"] = pd.to_datetime(tweet_info["time_stamp"])

            # tweet_info["time_stamp"] = datetime.strptime(tweet_info["time_stamp"],
            #                                              "%a %b %d %H:%M:%S +0000 %Y")

            all_frames_for_group = pd.merge(tweet_info, predictions, on="id_str")
            all_frames_for_group["group"] = "public"

        all_frame_dfs.append(all_frames_for_group)

        all_frames = pd.concat(all_frame_dfs)

    # get all of the valid frame labels
    all_frame_labels = []
    for frame_type in ["generic", "specific", "narrative"]:
        all_frame_labels.extend(config["frames"][frame_type])
    
    keep_cols = ["id_str", "time_stamp", "screen_name", "group"] + all_frame_labels

    return all_frames[keep_cols]

def filter_users_by_activity(tweet_df: pd.DataFrame, required_tweets: int) -> pd.DataFrame:
    return tweet_df.groupby("screen_name").filter(lambda x: len(x) > required_tweets)