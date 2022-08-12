import numpy as np
import pandas as pd


def make_handover_delay_results_table(RESULTS_DF: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the handover delay results table using the RESULTS_DF from the modelling,
    with the results showing as "values +/- error".
    Parameters
    ----------
    RESULTS_DF: pd.DataFrame
        Results df obtained from the model
    Return
    ------
    results_table: pd.DataFrame
        A df of the results table in the "values +/- error" format.
    """
    results_df = RESULTS_DF[['pred_3',
                             'mean_absolute_error_3',
                             'pred_10',
                             'mean_absolute_error_10',
                             'pred_24',
                             'mean_absolute_error_24']]
    # Setting the data types to "int" so that all the predictions are whole numbers
    results_df = results_df.astype({'pred_3': int,
                                    'pred_10': int,
                                    'pred_24': int,
                                    'mean_absolute_error_3': int,
                                    'mean_absolute_error_10': int,
                                    'mean_absolute_error_24': int})
    # Showing the handover time in the desired range format: "prediction +/- mean absolute error"
    results_df['3hr-Prediction (minutes)'] = results_df['pred_3'].astype(str) + ' ' + \
        u"\u00B1" + ' ' + results_df['mean_absolute_error_3'].astype(str)
    results_df['10hr-Prediction (minutes)'] = results_df['pred_10'].astype(str) + ' ' + \
        u"\u00B1" + ' ' + results_df['mean_absolute_error_10'].astype(str)
    results_df['24hr-Prediction (minutes)'] = results_df['pred_24'].astype(str) + ' ' + \
        u"\u00B1" + ' ' + results_df['mean_absolute_error_24'].astype(str)
    # Taking a subset of the relevant columns only
    results_table = results_df[['3hr-Prediction (minutes)',
                                '10hr-Prediction (minutes)',
                                '24hr-Prediction (minutes)']]
    # Renaming columns to more meaningful ones
    results_table = results_table.rename(columns={'3hr-Prediction (minutes)':
                                                  'Predicted Handover Delay in 3hr (minutes)',
                                                  '10hr-Prediction (minutes)':
                                                  'Predicted Handover Delay in 10hr (minutes)',
                                                  '24hr-Prediction (minutes)':
                                                  'Predicted Handover Delay in 24hr (minutes)'})
    # Some aesthetics
    results_table.index.name = 'Hospital'
    return results_table


def highlight_cells_depending_on_lower_bound(val):
    """
    Takes a scalar and returns a string with the css property:
    'background-color: green' for "lower_bound_values < 15",
    'background-color: amber' for "15 <= lower_bound_values < 30",
    'background-color: red' for "lower_bound_values >= 30"
    """
    # Using the lower bound
    val = float(val.split(' - ')[0])
    if val < 15:
        color = '#8CC690'
    elif val >= 15 and val < 30:
        color = '#FFBF00'
    else:
        color = '#FF7D7D'
    return 'background-color: %s' % color


def highlight_cells_depending_on_mean_of_lower_and_upper_bounds(val):
    """
    Takes a scalar and returns a string with the css property:
    'background-color: green' for "mean < 15",
    'background-color: amber' for "15 <= mean < 30",
    'background-color: red' for "mean >= 30"
    """
    # Splitting the range
    val_split = val.split(' - ')
    # First value => lower bound
    val1 = float(val_split[0])
    # Last value => upper bound
    val2 = float(val_split[-1])
    # Calculate the mean of the lower and upper bounds
    mean = np.mean([val1, val2])
    if mean < 15:
        color = '#8CC690'
    elif mean >= 15 and mean < 30:
        color = '#FFBF00'
    else:
        color = '#FF7D7D'
    return 'background-color: %s' % color


def make_handover_times_results_table(RESULTS_DF: pd.DataFrame,
                                      coloured_output: bool = True,
                                      highlight_by: str = 'mean') -> pd.DataFrame:
    """
    Creates the handover times results table using the RESULTS_DF from the modelling.
    Parameters
    ----------
    RESULTS_DF: pd.DataFrame
        Results df obtained from the model
    coloured_output: bool
        Whether cell highlighting is desired. Set to True by default.
    highlight_by: str
        Can be either of "mean" or "lower_bound", depending on the desired condition which
        is to be used for the cell highlighting.
    Return
    ------
    results_table: pd.DataFrame
        A df of the results table in the "lower_bound - upper_bound" format. Cells can be highlighted or not.
    """
    # Using a copy of the RESULTS_DF
    HANDOVER_TIMES_DF_ = RESULTS_DF.copy()
    # Col names to be used
    handover_time_cols = ['handover_time_3', 'handover_time_10', 'handover_time_24']
    handover_time_range_cols = ['handover_time_range_3hr', 'handover_time_range_10hr', 'handover_time_range_24hr']
    mean_absolute_error_cols = ['mean_absolute_error_3', 'mean_absolute_error_10', 'mean_absolute_error_24']
    lower_bound_cols = ['lower_bound_3', 'lower_bound_10', 'lower_bound_24']
    upper_bound_cols = ['upper_bound_3', 'upper_bound_10', 'upper_bound_24']
    # Using only a subset of the dataframe
    HANDOVER_TIMES_DF = HANDOVER_TIMES_DF_[['pred_3',
                                            'mean_absolute_error_3',
                                            'pred_10',
                                            'mean_absolute_error_10',
                                            'pred_24',
                                            'mean_absolute_error_24']]
    # Changing the "pred_" column names to "handover_time_" names
    HANDOVER_TIMES_DF = HANDOVER_TIMES_DF.rename(columns={'pred_3': 'handover_time_3',
                                                          'pred_10': 'handover_time_10',
                                                          'pred_24': 'handover_time_24'})
    # Add 15 minutes to the handover delay predictions to get the handover time
    for col in handover_time_cols:
        HANDOVER_TIMES_DF[col] = HANDOVER_TIMES_DF[col] + 15
    # Getting the lower and upper bounds of the prediction
    for enum, col_name in enumerate(lower_bound_cols):
        HANDOVER_TIMES_DF[col_name] = HANDOVER_TIMES_DF[handover_time_cols[enum]] \
            - HANDOVER_TIMES_DF[mean_absolute_error_cols[enum]]
        HANDOVER_TIMES_DF[upper_bound_cols[enum]] = HANDOVER_TIMES_DF[handover_time_cols[enum]] \
            + HANDOVER_TIMES_DF[mean_absolute_error_cols[enum]]
    # Changing the data type to "int" to make sure that they are round numbers
    HANDOVER_TIMES_DF = HANDOVER_TIMES_DF.astype({'handover_time_3': int,
                                                  'mean_absolute_error_3': int,
                                                  'handover_time_10': int,
                                                  'mean_absolute_error_10': int,
                                                  'handover_time_24': int,
                                                  'mean_absolute_error_24': int,
                                                  'lower_bound_3': int,
                                                  'upper_bound_3': int,
                                                  'lower_bound_10': int,
                                                  'upper_bound_10': int,
                                                  'lower_bound_24': int,
                                                  'upper_bound_24': int})
    # Converting negative time for the lower bounds to zero
    HANDOVER_TIMES_DF[HANDOVER_TIMES_DF < 0] = 0
    # Showing the handover time in the desired range format: "lower bound value - upper bound value"
    for enum, col_name in enumerate(handover_time_range_cols):
        HANDOVER_TIMES_DF[col_name] = HANDOVER_TIMES_DF[lower_bound_cols[enum]].astype(str) + \
            " - " + HANDOVER_TIMES_DF[upper_bound_cols[enum]].astype(str)
    # Including only a subset of the df in the final results table
    HANDOVER_TIMES_TABLE = HANDOVER_TIMES_DF[handover_time_range_cols]
    # Some aesthetics to the results table
    HANDOVER_TIMES_TABLE.index.name = 'Hospital'
    # Changing the column names to more representative ones
    HANDOVER_TIMES_TABLE = HANDOVER_TIMES_TABLE.rename(columns={'handover_time_range_3hr':
                                                                'Handover Time Range for 3hr (minutes)',
                                                                'handover_time_range_10hr':
                                                                'Handover Time Range for 10hr (minutes)',
                                                                'handover_time_range_24hr':
                                                                'Handover Time Range for 24hr (minutes)'})
    if coloured_output:
        # In case the highlighting condition is set to "mean"
        if "mean" in highlight_by:
            return HANDOVER_TIMES_TABLE.style.applymap(highlight_cells_depending_on_mean_of_lower_and_upper_bounds)
        # In case the highlighting condition is set to "lower_bound"
        elif "lower" in highlight_by:
            return HANDOVER_TIMES_TABLE.style.applymap(highlight_cells_depending_on_lower_bound)
    # Otherwise, no cell highlighting
    else:
        return HANDOVER_TIMES_TABLE


def make_feature_importance_table(RESULTS_DICT: dict, list_of_hospitals: list, keyword_to_english_map: dict):
    """
    Creates the feature importance table using the RESULTS_DICT from the modelling. Also maps the keywords obtained
    for the important features to their corresponding English words as descrived in the "keyword_to_english_map" dict.
    Parameters
    ----------
    RESULTS_DICT: dict
        The results dict obtained from the model.
    list_of_hospitals: list
        The list of all hospitals.
    keyword_to_english_map: dict
        A dict with the keywords as 'key', and their English equivalent as 'value'.
    Return
    ------
    important_features_df: pd.DataFrame
        A df containing all the important features of all the hospitals for all the time intervals.
    """
    # Make an empty df with 2 columns
    important_features_df = pd.DataFrame(columns=['Hospital', 'Feature Number'])
    # Set the 2 columns as indices
    important_features_df.set_index(['Hospital', 'Feature Number'], inplace=True)
    # The prediction columns
    pred_cols = ['3hr-Prediction', '10hr-Prediction', '24hr-Prediction']
    # Some counters
    col_count = 0
    hospital_counter = 0
    for k, v in RESULTS_DICT.items():
        # Checking if a key contains "impor" or not
        # if not, pass
        if "impor" not in k:
            pass
        # Otherwise, add the important features of each prediction interval to their corresponding hospital
        else:
            for hospital_features in v:
                for enum, feature in enumerate(hospital_features):
                    important_features_df.loc[(list_of_hospitals[hospital_counter], enum + 1), pred_cols[col_count]] \
                        = feature
                hospital_counter += 1
            col_count += 1
            hospital_counter = 0
    # Converting the 'keywords' which were obtained for important features to their corresponding 'english words'
    for col in important_features_df.columns:
        important_features_df[col] = important_features_df[col].map(keyword_to_english_map)
    return important_features_df
