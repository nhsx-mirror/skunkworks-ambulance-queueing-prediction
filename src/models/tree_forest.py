from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import BaseForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor, BaseDecisionTree


def prepare_train_test_split(df: pd.DataFrame):
    """
    Split the data into the training and testing sets for the feature and target variables.
    Parameters
    ----------
    df: pd.DataFrame
        It is the full input dataset which need to be split.
    Return
    ------
    feature_variables_train, feature_variables_test, target_variables_train, target_variables_test
        Subsets of the data for the feature and target variables
        which are to be used the training and testing process of the model.
    """
    # Dropping the target variable so that only the feature variables are left
    feature_variables = df.drop('handover_delay_mins', axis=1)
    # The target variables
    target_variables = df[['time', 'handover_delay_mins']]
    # Splitting the feature and target variables into the training and testing sets
    feature_variables_train, feature_variables_test, target_variables_train, target_variables_test = \
        train_test_split(feature_variables, target_variables, test_size=0.20, random_state=0)
    return feature_variables_train, feature_variables_test, target_variables_train, target_variables_test


def train_decision_tree_model(ORDINALS, REALS, target_df: pd.DataFrame, feature_df: pd.DataFrame) -> Pipeline:
    """
    Processes the data and trains the Decision Tree model.
    Parameters
    ----------
    ORDINALS: The categorical variables of the data.
    REALS: The continuous variables from the data.
    target_df: pd.DataFrame
        It is the df containing the target variable for training.
    feature_df: pd.dataFrame
        It is the df containing the feature variables for training.
    Return
    ------
    feature_subset: list
        The list of the feature variables that were used in the model.
    dtr: BaseForest
        Regressor variable for the decision tree.
    preprocessing: ColumnTransformer for the model input.
    pipeline: The fitted model.
    """
    # Takes the subset of the dataframe that contaions the feature variables
    feature_subset = [*ORDINALS.keys(), *REALS]
    # Ordinally encodes the categorical variables of the feature subset
    ordinal_encoder = OrdinalEncoder(categories=[v for k, v in ORDINALS.items() if k in feature_subset])
    # Converts missing values to zeros
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    # Construct a ColumnTransformer from the given transformers
    preprocessing = make_column_transformer(
        (ordinal_encoder, [k for k, v in ORDINALS.items() if k in feature_subset]),
        (simple_imputer, [k for k in REALS if k in feature_subset]))
    # Initialise the model
    dtr = DecisionTreeRegressor()
    # Build the pipeline
    pipeline = make_pipeline(preprocessing, dtr)
    # Fit the model
    pipeline.fit(feature_df[feature_subset], target_df)
    return feature_subset, dtr, preprocessing, pipeline


def tree_nodes_as_df(clf: BaseDecisionTree, id_offset=0, names=True):
    """
    Inspects the tree and outputs a dataframe of nodes of the tree describing the split points and their impurities.
    Parameters
    ----------
    clf: BaseDecisionTree
        The regressor variable for the decision tree.
    id_offset: int
        Set to zero as default.
    names: bool
        The feature names. Set to True as default.
    Return
    ------
    df: pd.DataFrame
        A df of all the nodes of the tree.
    """
    
    df = pd.DataFrame({'impurity': clf.tree_.impurity,
                       'child_left': clf.tree_.children_left,
                       'child_right': clf.tree_.children_right,
                       'feature': clf.feature_names_in_[clf.tree_.feature] if names else clf.tree_.feature,
                       'weighted_samples': clf.tree_.weighted_n_node_samples})\
        .rename_axis(index='node').reset_index()
    # Add left and right onto dfs below to know whether it came down left or right
    df = pd.concat([df.merge(df.add_prefix('parent_'), how='left', left_on='node', right_on=f'parent_child_{lr}')
                    for lr in ('left', 'right')])
    df['node'] = df['node'] + id_offset
    # deduplicate
    df = df[['node', 'impurity', 'parent_feature', 'weighted_samples', 'parent_impurity', 'parent_weighted_samples']]\
        .groupby('node').max()
    return df


def train_random_forest_model(ORDINALS, REALS, target_df: pd.DataFrame, df: pd.DataFrame) -> Pipeline:
    """
    Processes the data and trains the Random Forest model.
    Parameters
    ----------
    ORDINALS: The categorical variables of the data.
    REALS: The continuous variables from the data.
    target_df: pd.DataFrame
        It is the df containing the target variable for training.
    feature_df: pd.dataFrame
        It is the df containing the feature variables for training.
    Return
    ------
    feature_subset: list
        The list of the feature variables that were used in the model.
    dtr: BaseForest
        Regressor variable for the random forest.
    preprocessing: ColumnTransformer for the model input.
    pipeline: The fitted model.
    """
    # Takes the subset of the dataframe that contaions the feature variables
    feature_subset = [*ORDINALS.keys(), *REALS]
    # Encodes the categorical variables of the feature subset into numbers
    ordinal_encoder = OrdinalEncoder(categories=[v for k, v in ORDINALS.items() if k in feature_subset])
    # Converts missing values to zeros
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    # Construct a ColumnTransformer from the given transformers
    preprocessing = make_column_transformer(
        (ordinal_encoder, [k for k, v in ORDINALS.items() if k in feature_subset]),
        (simple_imputer, [k for k in REALS if k in feature_subset]))
    # Initialise the model
    rfr = RandomForestRegressor(oob_score=True, random_state=0, n_estimators=100)
    # Build the pipeline
    pipeline = make_pipeline(preprocessing, rfr)
    # Fit the model
    pipeline.fit(df[feature_subset], target_df['handover_delay_mins'])
    return feature_subset, rfr, preprocessing, pipeline


def forest_nodes_as_df(clf: BaseForest, names=True) -> pd.DataFrame:
    """
    Inspects the forest and outputs a dataframe of nodes of all the trees
    describing the split points and their impurities.
    Parameters
    ----------
    clf: BaseForest
        The regressor variable for the decision tree.
    names: bool
        The feature names. Set to True as default.
    Return
    ------
    df: pd.DataFrame
        A df of all the characteristics of the nodes.
    """
    # The node ID within each tree starts from 0
    # In output of decision path, the node ID is unique across the forest
    # Add the id_offset to make these match up
    id_offset = 0
    tree: BaseDecisionTree
    dfs = []
    # Loops through each of the trees inspecting their nodes
    for tree in clf.estimators_:
        dfs.append(tree_nodes_as_df(tree, id_offset=id_offset, names=False))
        id_offset += tree.tree_.node_count
    df = pd.concat(dfs)
    if names:
        df['feature'] = clf.feature_names_in_[df['feature']]
    return df


def loop_decision_tree_model(ORDINALS,
                             REALS,
                             datasets_dict: dict,
                             list_of_hospitals: list,
                             feature_importance: bool = True,
                             get_scatter_plots: bool = False):
    """
    Runs a Decision Tree model for each hospital, for each of the 3-, 10-, and 24-hours predictions.
    Parameters
    ----------
    ORDINALS: The categorical variables of the data.
    REALS: The continuous variables from the data.
    datasets_dict: dict
        A dict with 'str(hour)' as key and the corresponding df as value. Example below:
        datasets_dict = {'3': df_3,
                     '10': df_10,
                     '24': df_24}
    list_of_hospitals: list
        A list of all required hospital names
    feature_importance: bool
        Whether to extract the important features as well. Set to True by default.
    get_scatter_plots: bool
        Whether to print out the scatter plots. Set to False by default.
    Return
    ------
    RESULTS_DF: pd.DataFrame
        A df with all the prediction results and errors from the model.
    RESULTS_DICT: dict
        A dict of all the prediction results and the important features as well.
        """
    # Initialising empty results dict
    RESULTS_DICT = {'actual': [],
                    'predicted_3': [],
                    'predicted_10': [],
                    'predicted_24': [],
                    'important_features_3': [],
                    'important_features_10': [],
                    'important_features_24': [],
                    'important_features_3_value': [],
                    'important_features_10_value': [],
                    'important_features_24_value': []}
    # Initialising results df
    RESULTS_DF = pd.DataFrame()
    for hospital in list_of_hospitals:
        df_temp = pd.DataFrame()
        for hour, df_X in datasets_dict.items():
            print(f"Hospital: {hospital}, Hour: {hour}, Starting modelling...")
            # Take subset of data for the hospital for modelling
            hospital_subset = df_X[df_X.hospital == hospital]
            # Train-test split
            feature_variables_train, feature_variables_test, target_variables_train, target_variables_test = \
                prepare_train_test_split(hospital_subset)
            # Run model
            feature_subset, dtr, preprocessed, tree_model = \
                train_decision_tree_model(ORDINALS,
                                          REALS,
                                          target_variables_train,
                                          feature_variables_train[[*ORDINALS.keys(),
                                                                   *REALS]])
            # Get prediction
            target_variables_pred = tree_model.predict(feature_variables_test[[*ORDINALS.keys(), *REALS]])
            # Actual values
            target_variables_actual = target_variables_test.values
            # Add the results to the temporary table
            df_temp.loc[hospital, 'pred_' + hour] = target_variables_pred[-1]
            df_temp.loc[hospital, 'mean_absolute_error_' + hour] = mean_absolute_error(target_variables_pred,
                                                                                       target_variables_test)
            df_temp.loc[hospital, 'root_mean_squared_error_' + hour] = sqrt(mean_squared_error(target_variables_pred,
                                                                                               target_variables_test))
            # Codes to make scatter plot for y_pred and y_test (just to view how the scatter plots)
            if get_scatter_plots:
                fontsize = 16
                fig, ax = plt.subplots()
                ax.scatter(target_variables_test, target_variables_pred, alpha=0.5)
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                    ]
                # Now plot both limits against eachother
                ax.plot(lims, lims, '--', alpha=0.75, zorder=0, c='g')
                plt.title(f"{hospital} for {hour}-hours", fontsize=fontsize)
                plt.xlabel("Actual delays (minutes)", fontsize=fontsize)
                plt.ylabel("Predicted delays (minutes)", fontsize=fontsize)
                plt.xticks(fontsize=fontsize - 2, rotation=20)
                plt.yticks(fontsize=fontsize - 2, rotation=20)
                plt.show()
            # Update the results dict
            if hour == '3':
                RESULTS_DICT['actual'].append(target_variables_actual[-1])
            if hour == '3':
                RESULTS_DICT['predicted_3'].append(target_variables_pred[-1])
            elif hour == '10':
                RESULTS_DICT['predicted_10'].append(target_variables_pred[-1])
            elif hour == '24':
                RESULTS_DICT['predicted_24'].append(target_variables_pred[-1])
            if feature_importance:
                print(f"Hospital: {hospital}, Hour: {hour}, Getting important features...")
                print()
                # To get n important features: n=3 here
                RESULTS_DICT = get_important_features('decision tree',
                                                      RESULTS_DICT,
                                                      feature_variables_test,
                                                      ORDINALS,
                                                      REALS,
                                                      dtr,
                                                      preprocessed,
                                                      feature_subset,
                                                      3,
                                                      hour)
        # Update the results dataframe
        RESULTS_DF = RESULTS_DF.append(df_temp)
        RESULTS_DF = RESULTS_DF.round(2)
    return RESULTS_DF, RESULTS_DICT


def loop_random_forest_model(ORDINALS,
                             REALS,
                             datasets_dict: dict,
                             list_of_hospitals: list,
                             feature_importance: bool = True,
                             get_scatter_plots: bool = False):
    """
    Runs a Random Forest model for each hospital, for each of the 3-, 10-, and 24-hours predictions.
    Parameters
    ----------
    ORDINALS: The categorical variables of the data.
    REALS: The continuous variables from the data.
    datasets_dict: dict
        A dict with 'str(hour)' as key and the corresponding df as value. Example below:
        datasets_dict = {'3': df_3,
                     '10': df_10,
                     '24': df_24}
    list_of_hospitals: list
        A list of all required hospital names
    feature_importance: bool
        Whether to extract the important features as well. Set to True by default.
    get_scatter_plots: bool
        Whether to print out the scatter plots. Set to False by default.
    Return
    ------
    RESULTS_DF: pd.DataFrame
        A df with all the prediction results and errors from the model.
    RESULTS_DICT: dict
        A dict of all the prediction results and the important features as well.
    """
    # Initialising empty results dict
    RESULTS_DICT = {'actual': [],
                    'predicted_3': [],
                    'predicted_10': [],
                    'predicted_24': [],
                    'important_features_3': [],
                    'important_features_10': [],
                    'important_features_24': []}
    # Initialising results df
    RESULTS_DF = pd.DataFrame()
    for hospital in list_of_hospitals:
        df_temp = pd.DataFrame()
        for hour, df_X in datasets_dict.items():
            print(f"Hospital: {hospital}, Hour: {hour}, Starting modelling...")
            # Take subset of data for the hospital for modelling
            hospital_subset = df_X[df_X.hospital == hospital]
            # Train-test split
            feature_variables_train, feature_variables_test, target_variables_train, target_variables_test = \
                prepare_train_test_split(hospital_subset)
            # Run model
            feature_subset, rfr, preprocessed, forest_model = \
                train_random_forest_model(ORDINALS,
                                          REALS,
                                          target_variables_train,
                                          feature_variables_train[[*ORDINALS.keys(),
                                                                   *REALS]])
            # Get prediction
            # Using training data
            target_variables_pred_train_data = forest_model.predict(feature_variables_train[[*ORDINALS.keys(), *REALS]])
            # Using test data
            target_variables_pred_test_data = forest_model.predict(feature_variables_test[[*ORDINALS.keys(), *REALS]])
            # Actual values
            target_variables_actual = target_variables_test.values
            df_temp.loc[hospital, 'pred_' + hour] = target_variables_pred_test_data[-1]
            df_temp.loc[hospital, 'mean_absolute_error_' + hour] = mean_absolute_error(target_variables_pred_test_data,
                                                                                       target_variables_test['handover_delay_mins'])
            df_temp.loc[hospital, 'root_mean_squared_error_' + hour] = sqrt(mean_squared_error(target_variables_pred_test_data,
                                                                                               target_variables_test['handover_delay_mins']))
            
            # Getting the daily dfs
            # For test data
            target_variables_test['prediction_test'] = target_variables_pred_test_data
            target_variables_test_daily = target_variables_test.groupby(target_variables_test.time.dt.floor('d')).mean()
            
            # For train data
            target_variables_train['prediction_train'] = target_variables_pred_train_data
            target_variables_train_daily = target_variables_train.groupby(target_variables_train.time.dt.floor('d')).mean()
            
            
            # Codes to make scatter plot for y_pred and y_test 
            if get_scatter_plots:
                fontsize = 16
                # print("The scatter plot for the predictions from the testing data:")
                # fig, ax = plt.subplots(figsize=(10, 8))
                # ax.scatter(target_variables_test['handover_delay_mins'], target_variables_pred_test_data, alpha=0.5)
                # lims = [
                #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                #     ]
                # # Now plot both limits against eachother
                # ax.plot(lims, lims, '--', alpha=0.75, zorder=0, c='g')
                # plt.title(f"{hospital} {hour}-hour Predictions (Testing Data)", fontsize=fontsize)
                # plt.xlabel("Actual Delays (minutes)", fontsize=fontsize)
                # plt.ylabel("Predicted Delays (minutes)", fontsize=fontsize)
                # plt.xticks(fontsize=fontsize - 2, rotation=20)
                # plt.yticks(fontsize=fontsize - 2, rotation=20)
                # plt.show()
                
                # print("The scatter plot for the predictions from the training data:")
                # fig, ax = plt.subplots(figsize=(10, 8))
                # ax.scatter(target_variables_train['handover_delay_mins'], target_variables_pred_train_data, alpha=0.5)
                # lims = [
                #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                #     ]
                # # Now plot both limits against eachother
                # ax.plot(lims, lims, '--', alpha=0.75, zorder=0, c='g')
                # plt.title(f"{hospital} {hour}-hour Predictions (Training Data)", fontsize=fontsize)
                # plt.xlabel("Actual delays (minutes)", fontsize=fontsize)
                # plt.ylabel("Predicted delays (minutes)", fontsize=fontsize)
                # plt.xticks(fontsize=fontsize - 2, rotation=20)
                # plt.yticks(fontsize=fontsize - 2, rotation=20)
                # plt.show()
                
                print("The scatter plot for the daily mean predictions from the testing data:")
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(target_variables_test_daily['handover_delay_mins'], target_variables_test_daily['prediction_test'], alpha=0.5)
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                    ]
                # Now plot both limits against eachother
                ax.plot(lims, lims, '--', alpha=0.75, zorder=0, c='g')
                plt.title(f"{hospital} {hour}-hour Predictions (Testing Data)", fontsize=fontsize)
                plt.xlabel("Average Daily Actual Delays (minutes)", fontsize=fontsize)
                plt.ylabel("Average Daily Predicted Delays (minutes)", fontsize=fontsize)
                plt.xticks(fontsize=fontsize - 2, rotation=20)
                plt.yticks(fontsize=fontsize - 2, rotation=20)
                plt.show()
                
                # print("The scatter plot for the daily mean predictions from the training data:")
                # fig, ax = plt.subplots(figsize=(10, 8))
                # ax.scatter(target_variables_train_daily['handover_delay_mins'], target_variables_train_daily['prediction_train'], alpha=0.5)
                # lims = [
                #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                #     ]
                # # Now plot both limits against eachother
                # ax.plot(lims, lims, '--', alpha=0.75, zorder=0, c='g')
                # plt.title(f"{hospital} {hour}-hour Predictions (Training Data)", fontsize=fontsize)
                # plt.xlabel("Average Daily Actual Delays (minutes)", fontsize=fontsize)
                # plt.ylabel("Average Daily Predicted Delays (minutes)", fontsize=fontsize)
                # plt.xticks(fontsize=fontsize - 2, rotation=20)
                # plt.yticks(fontsize=fontsize - 2, rotation=20)
                # plt.show()
                
            # Update the results dict
            if hour == '3':
                RESULTS_DICT['actual'].append(target_variables_actual[-1])
            if hour == '3':
                RESULTS_DICT['predicted_3'].append(target_variables_pred_test_data[-1])
            elif hour == '10':
                RESULTS_DICT['predicted_10'].append(target_variables_pred_test_data[-1])
            elif hour == '24':
                RESULTS_DICT['predicted_24'].append(target_variables_pred_test_data[-1])
            if feature_importance:
                print(f"Hospital: {hospital}, Hour: {hour}, Getting important features...")
                print()
                # to get n most important features: n=3 here
                RESULTS_DICT = get_important_features('random forest',
                                                      RESULTS_DICT,
                                                      feature_variables_test,
                                                      ORDINALS,
                                                      REALS,
                                                      rfr,
                                                      preprocessed,
                                                      feature_subset,
                                                      3,
                                                      hour)
        # Update the results dataframe
        RESULTS_DF = RESULTS_DF.append(df_temp)
        RESULTS_DF = RESULTS_DF.round(2)
    return RESULTS_DF, RESULTS_DICT


def get_important_features(model_name: str,
                           RESULTS_DICT: dict,
                           feature_variables_test: pd.DataFrame,
                           ORDINALS,
                           REALS,
                           regressor_variable,
                           preprocessed_variable,
                           feature_subset,
                           n_important_features: int,
                           hour: str):
    """
    Extracts the most important features from the model, and adds the important features to the the results dict.
    Parameters
    ----------
    model_name: str
        The name of the model for which it is being used. Can be either of "random_forest" or "decision_tree".
    RESULTS_DICT: dict
        The dict used to store all the results.
    feature_variables_test: pd.DataFrame
        The df containing the feature variables that were used in the testing process of the model.
    ORDINALS: The categorical variables of the data.
    REALS: The continuous variables from the data.
    regressor_variable: The regressor variable from the model. Can be either of "dtr" or "rfr".
    preprocessed_variable: ColumnTransformer for the model input.
    feature_subset: list
        The list of the feature variables that were used in the model.
    n_important_features: int
        The number of important features to extract.
    hour: str
        The time interval for which the model is being run. It is used to save the results in the correct place.
    Return
    ------
    RESULTS_DICT: dict
        The updated result dict with the important features included.
    """
    if 'forest' in model_name.lower().replace(' ', ''):
        dp, _ = regressor_variable.decision_path(preprocessed_variable.
                                                 transform(feature_variables_test[[*ORDINALS, *REALS]]))
    elif 'tree' in model_name.lower().replace(' ', ''):
        dp = regressor_variable.decision_path(preprocessed_variable.
                                              transform(feature_variables_test[[*ORDINALS, *REALS]]))
    dp = pd.MultiIndex.from_arrays(dp.nonzero(), names=['observation', 'node']).to_frame()[[]]
    if 'forest' in model_name.lower().replace(' ', ''):
        df = forest_nodes_as_df(regressor_variable, names=False).join(dp)
    elif 'tree' in model_name.lower().replace(' ', ''):
        # Joining the tree nodes to the decision path
        df = tree_nodes_as_df(regressor_variable, names=False).join(dp)
    # Calculating the feature importance
    df['feature_importance'] = (df['parent_impurity'] - df['impurity']) * df['weighted_samples']
    df_features = pd.DataFrame([enum, feature] for enum, feature in enumerate(feature_subset)) \
        .rename(columns={0: 'parent_feature',
                         1: 'parent_feature_name'})
    df = df.reset_index().merge(df_features, on='parent_feature', how='left')
    df_grouped = df.groupby(['observation', 'parent_feature_name']).sum().sort_values(['observation',
                                                                                       'feature_importance'],
                                                                                      ascending=[True, False])
    # n most important features: n=3 (Selecting the three features that cause the largest decrease in impurity)
    n_largest = df_grouped.groupby('observation')['feature_importance'].nlargest(n_important_features)
    important_feature_df = n_largest.iloc[-3:].to_frame()
    important_features = np.array([important_feature_df.index[i][2] for i, v in enumerate(important_feature_df.index)])
    if hour == '3':
        RESULTS_DICT['important_features_3'].append(important_features)
    elif hour == '10':
        RESULTS_DICT['important_features_10'].append(important_features)
    elif hour == '24':
        RESULTS_DICT['important_features_24'].append(important_features)
    return RESULTS_DICT
