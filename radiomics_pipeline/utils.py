import sklearn
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


############ pre-process feature table ######################

def get_correlated_features_to_drop(thres_dataset_train):
    cor = thres_dataset_train.corr('spearman').abs()
    upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
    to_drop = []
    for column in upper_tri.columns:
        for row in upper_tri.columns:
            if upper_tri[column][row] > 0.85:
                if np.sum(upper_tri[column]) > np.sum(
                        upper_tri[row]):
                    to_drop.append(column)
                else:
                    to_drop.append(row)
    to_drop = np.unique(to_drop)
    return to_drop


def preprocessing_train(df_true_mask_train_features):  ##patient name needs to be removed
    ##normalize the features
    mean_std = {}
    for var in df_true_mask_train_features.columns:
        temp_mean = df_true_mask_train_features[var].mean()
        temp_std = df_true_mask_train_features[var].std()
        mean_std[var] = (temp_mean, temp_std)
        df_true_mask_train_features[var] = (df_true_mask_train_features[var] - temp_mean) / temp_std
    ##remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df_true_mask_train_features)
    thres_dataset_train = df_true_mask_train_features.loc[:, selector.get_support()]
    ## get_correlated_features_to_drop
    to_drop = get_correlated_features_to_drop(thres_dataset_train)
    decor_dataset = thres_dataset_train.drop(to_drop, axis=1)
    return mean_std, selector, to_drop, decor_dataset


def preprocessing_test(df_true_mask_test_features, mean_std, selector, to_drop):  ##apply parameters to test dataset
    for var in df_true_mask_test_features.columns:
        df_true_mask_test_features[var] = (df_true_mask_test_features[var] - mean_std[var][0]) / mean_std[var][1]
    thres_dataset_test = df_true_mask_test_features.loc[:, selector.get_support()]
    decor_dataset_test = thres_dataset_test.drop(to_drop, axis=1)
    return decor_dataset_test


##################### generate results #################


def get_optimal_threshold(true_outcome, predictions):
    ##to obtain a good threshold based on the train dataset
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_outcome, predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def get_results(y_label, y_pred, label,
                optimal_threshold):  # better function below to get results with confidence interval
    ##optimal threshold: reuse the one computed on the train dataset
    ##label: index of the dataframe, can be "external radiomics results"
    ##returns a dataframe with auc accuracy precision recall f1-score
    dict_results = {}
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_label, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    dict_results["auc"] = roc_auc
    y_pred_binary = (np.array(y_pred) > optimal_threshold).astype(int)
    dict_results["accuracy"] = [sklearn.metrics.accuracy_score(y_label, y_pred_binary)]
    dict_results["precision"] = [sklearn.metrics.precision_score(y_label, y_pred_binary)]
    dict_results["recall"] = [sklearn.metrics.recall_score(y_label, y_pred_binary)]
    dict_results["f1 score"] = [sklearn.metrics.f1_score(y_label, y_pred_binary)]
    df_results = pd.DataFrame.from_dict(dict_results)
    df_results = df_results.reset_index(drop=True)
    df_results.index = [label]
    return df_results


np.random.seed(32)


def bootstrap(label, pred, f, nsamples=2000):
    stats = []
    for b in range(nsamples):
        random_list = np.random.randint(label.shape[0], size=label.shape[0])
        stats.append(f(label[random_list], pred[random_list]))
    return stats, np.percentile(stats, (2.5, 97.5))


def nom_den(label, pred, f):
    if f == sklearn.metrics.accuracy_score:
        n = np.sum(label == pred)
        d = len(pred)
    if f == sklearn.metrics.precision_score:
        n = np.sum(pred[label == 1])
        d = np.sum(pred)
    if f == sklearn.metrics.recall_score:
        n = np.sum(pred[label == 1])
        d = np.sum(label)
    if f == sklearn.metrics.f1_score:
        n = 0
        d = 0
    return n, d


def get_ci(label, pred, f):
    stats, ci = bootstrap(label, pred, f)
    n, d = nom_den(label, pred, f)
    return stats, ["%5d/%5d (%5d %% )  CI [%0.2f,%0.2f]" % (
    n, d, int(f(label, pred) * 100), ci[0], ci[1])]  # doesn't compute the mean of the score


def get_ci_for_auc(label, pred, nsamples=2000):
    auc_values = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for b in range(nsamples):
        idx = np.random.randint(label.shape[0], size=label.shape[0])
        temp_pred = pred[idx]
        temp_fpr, temp_tpr, temp_thresholds = sklearn.metrics.roc_curve(label[idx], temp_pred)
        roc_auc = sklearn.metrics.auc(temp_fpr, temp_tpr)
        auc_values.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, temp_fpr, temp_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ci_auc = np.percentile(auc_values, (2.5, 97.5))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pred)
    return auc_values, ["%0.2f CI [%0.2f,%0.2f]" % (sklearn.metrics.auc(fpr, tpr), ci_auc[0], ci_auc[1])]


def get_stats_with_ci(y_label, y_pred, label, optimal_threshold, nsamples=2000):
    ##optimal threshold: reuse the one computed on the train dataset
    ##label: index of the dataframe, can be "external radiomics results"
    ##returns a dataframe with auc accuracy precision recall f1-score
    dict_results = {}
    dict_distributions = {}
    dict_distributions["auc"], dict_results["auc"] = get_ci_for_auc(y_label, y_pred)
    y_pred_binary = (np.array(y_pred) > optimal_threshold).astype(int)
    dict_distributions["accuracy"], dict_results["accuracy"] = get_ci(y_label, y_pred_binary,
                                                                      sklearn.metrics.accuracy_score)
    dict_distributions["precision"], dict_results["precision"] = get_ci(y_label, y_pred_binary,
                                                                        sklearn.metrics.precision_score)
    dict_distributions["specificity"], dict_results["specificity"] = get_ci(np.ones(len(y_label)) - y_label,
                                                                            np.ones(len(y_pred_binary)) - y_pred_binary,
                                                                            sklearn.metrics.recall_score)
    dict_distributions["recall"], dict_results["recall"] = get_ci(y_label, y_pred_binary, sklearn.metrics.recall_score)
    dict_distributions["f1 score"], dict_results["f1 score"] = get_ci(y_label, y_pred_binary, sklearn.metrics.f1_score)
    df_results = pd.DataFrame.from_dict(dict_results)
    df_results = df_results.reset_index(drop=True)
    df_results.index = [label]
    df_distributions = pd.DataFrame.from_dict(dict_distributions)
    df_distributions = df_distributions.reset_index(drop=True)
    return df_distributions, df_results


######################### save and load model and results ############################

def save_all_params(path_to_save, rfe, filtered_col, gsearch, mean_std, to_drop, selector, support, proba_train,
                    proba_test, proba_external, data_used="radiomics"):
    ##give the path to save
    filename_filtered_col = path_to_save + 'filtered_col_' + data_used + '.pkl'
    pickle.dump(rfe, open(filename_filtered_col, 'wb'))
    filename_rfe = path_to_save + 'rfe_' + data_used + '.pkl'
    pickle.dump(rfe, open(filename_rfe, 'wb'))
    filename_gsearch = path_to_save + 'gsearch_' + data_used + '.pkl'
    pickle.dump(gsearch, open(filename_gsearch, 'wb'))
    filename_parameters = path_to_save + r"parameters_" + data_used + ".pkl"
    pickle.dump([mean_std, selector, to_drop, support], open(filename_parameters, 'wb'))
    filename_proba_train = path_to_save + r"proba_train_" + data_used + ".pkl"
    pickle.dump(proba_train, open(filename_proba_train, 'wb'))
    filename_proba_test = path_to_save + r"proba_test_" + data_used + ".pkl"
    pickle.dump(proba_test, open(filename_proba_test, 'wb'))
    filename_proba_external = path_to_save + r"proba_external_" + data_used + ".pkl"
    pickle.dump(proba_external, open(filename_proba_external, 'wb'))
    return "done"


def load_all_params(path_to_load, data_used="radiomics"):
    ##give the path to load
    filename_filtered_col = path_to_load + 'filtered_col_' + data_used + '.pkl'
    filtered_col = pickle.load(open(filename_filtered_col, 'rb'))
    filename_rfe = path_to_load + 'rfe_' + data_used + '.pkl'
    rfe = pickle.load(open(filename_rfe, 'rb'))
    filename_gsearch = path_to_load + 'gsearch_' + data_used + '.pkl'
    gsearch = pickle.load(open(filename_gsearch, 'rb'))
    filename_parameters = path_to_load + r"parameters_" + data_used + ".pkl"
    [mean_std, selector, to_drop, support] = pickle.load(open(filename_parameters, 'rb'))
    filename_proba_train = path_to_load + r"proba_train_" + data_used + ".pkl"
    proba_train = pickle.load(open(filename_proba_train, 'rb'))
    filename_proba_test = path_to_load + r"proba_test_" + data_used + ".pkl"
    proba_test = pickle.load(open(filename_proba_test, 'rb'))
    filename_proba_external = path_to_load + r"proba_external_" + data_used + ".pkl"
    proba_external = pickle.load(open(filename_proba_external, 'rb'))
    return rfe, filtered_col, gsearch, mean_std, to_drop, selector, support, proba_train, proba_test, proba_external