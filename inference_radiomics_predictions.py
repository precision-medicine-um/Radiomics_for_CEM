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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference code for training a ML model with radiomics')
    parser.add_argument('--path_csv', default=None,  help='Path to the csv containing mask/low_energy/recombined paths')
    parser.add_argument('--path_to_load_parameters', default=None, help='path to load the parameters')
    parser.add_argument('--path_to_load_model', default=None, help='path to load the model')
    parser.add_argument('--path_dir_to_save', help='Path to the directory to save the results')

    args = parser.parse_args()

    #generate radiomics features
    if args.path_csv is not None:
        extractor = initialize_feature_extractor()
        df = pd.read_csv(args.path_csv)
        df_features = generate_features_table(df, extractor,inference_usage=True)

    #load parameters
    filename_parameters = args.path_to_load_parameters
    with open(filename_parameters, 'rb') as f:
        mean_std, selector, to_drop, support = pickle.load(f)
    path_images = list(df_features.index)
    df_features.reset_index(drop=True,inplace=True)
    decor_dataset = preprocessing_test(df_features, mean_std, selector, to_drop)

    #use the selected features only
    filtered_col = np.extract(support, np.array(decor_dataset.columns))
    reduced_features = decor_dataset[filtered_col]
    print("features processed")

    #load model
    filename_gsearch = args.path_to_load_model
    with open(filename_parameters, 'rb') as f:
        gsearch = pickle.load(f)
    best_estimator = gsearch.best_estimator_

    #predict case by case
    for i,index in enumerate(path_images):
        temp_proba = best_estimator.predict_proba(reduced_features[i])
        file = open(args.path_dir_to_save + "/" + ntpath.basename(index).split(".")[0]+".pkl", "wb")
        pickle.dump(temp_proba, file)
        file.close()

    print("Radiomics predictions generated for " + str(i+1) + " images as pickle")
