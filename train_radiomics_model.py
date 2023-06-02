import os
import pickle
import pandas as pd
import argparse
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from preprocessing.radiomics_extraction import initialize_feature_extractor,generate_features_table
from radiomics_pipeline.utils import preprocessing_train, preprocessing_test

#todo: add clinical feature option
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference code for training a ML model with radiomics')
    parser.add_argument('--path_csv_train', default=None,  help='Path to the csv containing mask/low_energy/recombined paths')
    parser.add_argument('--path_csv_test', default=None, help='Path to the csv containing mask/low_energy/recombined paths')
    parser.add_argument('--path_dir_features_to_save', required=True, help='path to save radiomics features')
    parser.add_argument('--find_optimal_features', default=False, help='train a RFECV to find the optimal features')
    parser.add_argument('--path_to_save_parameters', default=None, help='path to store the best model and the predictions')
    args = parser.parse_args()

    #generate radiomics features
    if args.path_csv_train is not None:
        extractor = initialize_feature_extractor()
        df_train = pd.read_csv(args.path_csv_train)
        df_features_train = generate_features_table(df_train, extractor)
        df_features_train.to_csv(os.path.join(args.path_dir_features_to_save,"df_features_train.csv"))
        print("radiomics features saved for the train set")
    else:
        raise Exception("you must provide a path to csv train")

    if args.path_csv_test is not None:
        extractor = initialize_feature_extractor()
        df_test = pd.read_csv(args.path_csv_test)
        df_features_test = generate_features_table(df_test, extractor)
        df_features_test.to_csv(os.path.join(args.path_dir_features_to_save,"df_features_test.csv"))
        print("radiomics features saved for the test set")
    else:
        raise Exception("you must provide a path to csv test")

    #load features
    df_features_train = pd.read_csv(os.path.join(args.path_dir_features_to_save,"df_features_train.csv"))
    outcome_train = list(df_features_train["outcome"])
    df_features_train.drop(["Unnamed: 0","outcome"], inplace=True, axis=1)
    df_features_test = pd.read_csv(os.path.join(args.path_dir_features_to_save, "df_features_test.csv"))
    outcome_test = list(df_features_test["outcome"])
    df_features_test.drop(["Unnamed: 0","outcome"], inplace=True, axis=1)

    #preprocess features
    mean_std, selector, to_drop, decor_dataset_train = preprocessing_train(df_features_train)
    decor_dataset_test = preprocessing_test(df_features_test, mean_std, selector, to_drop)
    print("features processed")

    #select optimal features
    model = xgb.XGBClassifier(use_label_encoder=False, colsample_bytree=1,
                              objective='binary:logistic', eval_metric='logloss', nthread=4, scale_pos_weight=1,
                              seed=27)

    if args.find_optimal_features:
        min_features_to_select = 1  # Minimum number of features to consider
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10),
                      scoring='roc_auc',
                      min_features_to_select=min_features_to_select)
        rfecv.fit(decor_dataset_train, outcome_train)
        support = rfecv.support_

    else:
        rfe = RFE(estimator=model, n_features_to_select=10)
        rfe.fit(decor_dataset_train, outcome_train)
        support = rfe.support_

    filtered_col = np.extract(support, np.array(decor_dataset_train.columns))
    reduced_features_train_set = decor_dataset_train[filtered_col]

    param_test_xgb = {
        'max_depth': range(2, 4, 1),
        'min_child_weight': range(1, 6, 2),
        'gamma': [i * 0.1 for i in range(1, 10)],
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=1000, num=10)],
        'learning_rate': [10 ** (-i) for i in range(2, 7)]
    }

    kfold = StratifiedKFold(n_splits=5, random_state=42)
    gsearch = GridSearchCV(model,
                           param_grid=param_test_xgb, scoring='roc_auc', n_jobs=6, iid=False, cv=kfold, verbose=1)

    gsearch.fit(reduced_features_train_set, outcome_train)

    reduced_features_test_set = decor_dataset_test[filtered_col]

    best_estimator = gsearch.best_estimator_

    proba_train = best_estimator.predict_proba(reduced_features_train_set)
    proba_test = best_estimator.predict_proba(reduced_features_test_set)

    ##save everything
    if args.path_to_save_parameters is not None:

        path_to_save = args.path_to_save_parameters

        if args.find_optimal_features:
            filename_rfecv = path_to_save +'rfecv_radiomics.pkl'
            pickle.dump(rfecv, open(filename_rfecv, 'wb'))
        else:
            filename_rfe = path_to_save +'rfe_radiomics.pkl'
            pickle.dump(rfe, open(filename_rfe, 'wb'))

        filename_gsearch = path_to_save +'gsearch_radiomics.pkl'
        pickle.dump(gsearch, open(filename_gsearch, 'wb'))
        filename_parameters =path_to_save + r"parameters_radiomics.pkl"
        pickle.dump([mean_std,selector, to_drop,support],open(filename_parameters, 'wb'))
        filename_proba_train =path_to_save + r"proba_train_radiomics.pkl"
        pickle.dump(proba_train,open(filename_proba_train, 'wb'))
        filename_proba_test =path_to_save + r"proba_test_radiomics.pkl"
        pickle.dump(proba_test,open(filename_proba_test, 'wb'))
