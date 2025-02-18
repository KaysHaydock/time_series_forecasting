import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Helper function to convert predictions to a pandas DataFrame
def preds_as_series(data, preds):
    """
    Helper function for plotting predictions.
    Converts a numpy array of predictions to a
    pandas.DataFrame with datetimeindex

    Parameters
    -----
    data : arraylike
        The training data used to determine the index frequency
    preds : numpy.array
        The vector of predictions

    Returns:
    -------
    pandas.DataFrame
    """
    start = pd.date_range(start=data.index.max(),
                          periods=2,
                          freq=data.index.freq).max()

    idx = pd.date_range(start=start,
                        periods=len(preds),
                        freq=data.index.freq)

    return pd.DataFrame(preds, index=idx)


# Helper function to help plot prediction intervals
def plot_prediction_intervals(train, preds, intervals, test=None, title=None):
    """
    Helper function to plot training data, point preds
    and 2 sets of prediction intevals (PI)

    This function assumes two sets of PIs are provided

    Parameters
    ----------
    train : pandas.DataFrame
        The training data to be plotted

    preds : numpy.ndarray
        The predictions

    intervals : list of numpy.ndarray
        A list containing two arrays of prediction intervals
        (lower and upper bounds)

    test : pandas.DataFrame, optional
        The test data to plot on the graph - Defaults to None

    title : str, optional
        The title to be set for the plot - Defaults to None
    """
    ax = train.plot(figsize=(12, 4), style="-", color="black")

    mean = preds_as_series(train, preds)
    intervals_80 = preds_as_series(train, intervals[0])
    intervals_90 = preds_as_series(train, intervals[1])

    mean.plot(ax=ax, label="point forecast", color="blue")

    ax.fill_between(
        intervals_80.index,
        mean[0],
        intervals_80[1],
        alpha=0.2,
        label="80% PI",
        color="orange",
    )

    ax.fill_between(
        intervals_80.index,
        mean[0],
        intervals_80[0],
        alpha=0.2,
        label="80% PI",
        color="orange",
    )

    ax.fill_between(
        intervals_80.index,
        intervals_80[1],
        intervals_90[1],
        alpha=0.2,
        label="90% PI",
        color="blue",
    )

    ax.fill_between(
        intervals_80.index,
        intervals_80[0],
        intervals_90[0],
        alpha=0.2,
        label="90% PI",
        color="blue",
    )

    ax.set_ylabel("Number of Attendances")

    if test is None:
        ax.legend(
            ["train", "point forecast", "80%PI",
             "90%PI"], loc=2
        )
        plt.grid(True, alpha=0.2)
    else:
        test.plot(ax=ax, color="red", linestyle='dashed', label="Test")
        ax.legend(
            ["train", "point forecast", "Test", "80%PI",
             "90%PI"], loc=2,
        )
        plt.grid(True, alpha=0.2)

    if title:
        ax.set_title(title)


# Helper function to help convert arima preds into a dataframe        
def arima_preds_to_dataframe(y_train, preds, intervals):
    """
    pdmarima.ARIMA models return point forecasts and prediction
    intervals as numpy arrays.
    
    This function converts them into data frames with a
    DateTimeIndex.

    Parameters:
    ---------
    y_train: pd.DataFrame or pd.Series
        Training data

    preds: np.array
        point forecasts

    intervals: np.array
        matrix - prediction intervals

    Returns:
    -------
        pd.DataFrame['mean'], pd.DataFrame['lower', 'upper']

    """
    pred_idx = pd.date_range(
        start=y_train.index[-1],
        periods=len(preds)+1,
        freq=y_train.index.freq
    )[1:]

    preds = pd.DataFrame(preds, index=pred_idx)
    preds.columns = ['mean']

    intervals = pd.DataFrame(intervals, index=pred_idx)
    intervals.columns = ['lower', 'upper']

    return preds, intervals


# Prophet requires the data within a DataFrame with 2 x columns 'ds' & y
def prophet_training_data(y_train):
    """
    Converts a standard pandas datetimeindexed dataframe
    for time series into one suitable for Prophet

    Parameters:
    ---------
    y_train: pd.DataFrame
        The univariate time series data

    Returns:
    --------
    pd.DataFrame 
        A DataFrame in Prophet format
        columns = ['ds', 'y']
    """
    prophet_train = pd.DataFrame(y_train.index)
    prophet_train['y'] = y_train.to_numpy()
    prophet_train.columns = ['ds', 'y']

    return prophet_train        