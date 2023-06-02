from radiomics import featureextractor
import pandas as pd
import tqdm



def initialize_feature_extractor():
    paramsFile = "CEM_extraction.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile, shape2D=True, force2D=True,
                                                           force2Ddimension=True, resampledPixelSpacing=None)
    extractor.addProvenance(False)
    extractor.disableAllFeatures()
    extractor.enableImageTypes(Original={})

    extractor.enableFeatureClassByName('firstorder', enabled=True)
    extractor.enableFeatureClassByName('shape2D', enabled=True)
    extractor.enableFeatureClassByName('glcm', enabled=True)
    extractor.enableFeatureClassByName('glrlm', enabled=True)
    extractor.enableFeatureClassByName('glszm', enabled=True)
    extractor.enableFeatureClassByName('gldm', enabled=True)
    extractor.enableFeatureClassByName('ngtdm', enabled=True)
    return extractor


def generate_features_table(df, extractor,inference_usage=False):
    # warning: this function can take a long time to run
    # extract low energy features
    featureVector_low_energy = extractor.execute(list(df["path_low_energy"])[0], list(df["path_mask"])[0])
    temp_dataset = pd.Series(featureVector_low_energy)
    feature_df_low_energy = pd.DataFrame([temp_dataset], columns=list(featureVector_low_energy.keys()),
                                         index=[list(df["path_mask"])])
    for i, temp_mask in tqdm.tqdm(enumerate(list(df["path_mask"])[1:])):
        featureVector_low_energy = extractor.execute(list(df["path_low_energy"])[i + 1], temp_mask)
        temp_dataset = pd.Series(featureVector_low_energy)
        feature_df_low_energy.loc[temp_mask] = temp_dataset.values
    feature_df_low_energy.columns = feature_df_low_energy.columns + "_low_energy"
    # extract recombined features
    featureVector_recombined = extractor.execute(list(df["path_recombined"])[0], list(df["path_mask"])[0])
    temp_dataset = pd.Series(featureVector_recombined)
    feature_df_recombined = pd.DataFrame([temp_dataset], columns=list(featureVector_recombined.keys()),
                                         index=[list(df["path_mask"])])
    for i, temp_mask in tqdm.tqdm(enumerate(list(df["path_mask"])[1:])):
        featureVector_recombined = extractor.execute(list(df["path_recombined"])[i + 1], temp_mask)
        temp_dataset = pd.Series(featureVector_recombined)
        feature_df_recombined.loc[temp_mask] = temp_dataset.values
    feature_df_recombined.columns = feature_df_recombined.columns + "_recombined"
    if not inference_usage:
        feature_df_recombined["outcome"] = df["outcome"].values
    return feature_df_recombined.join(feature_df_low_energy)