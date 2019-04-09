import os
import pandas as pd
import json
from sklearn.externals import joblib
from django.core.cache import cache


def get_fullset_of_features():
    """
    Gets the full set of features for the final hot encoded data
    :return: A empty data frame with all the features needed for prediction
    """
    hot_encoded_data = {
        'age': [0],
        'fnlwgt': [0],
        'capital-gain': [0],
        'capital-loss': [0],
        'hours-per-week': [0],
        'workclass_?': [0],
        'workclass_Federal-gov': [0],
        'workclass_Local-gov': [0],
        'workclass_Never-worked': [0],
        'workclass_Private': [0],
        'workclass_Self-emp-inc': [0],
        'workclass_Self-emp-not-inc': [0],
        'workclass_State-gov': [0],
        'workclass_Without-pay': [0],
        'education_10th': [0],
        'education_11th': [0],
        'education_12th': [0],
        'education_1st-4th': [0],
        'education_5th-6th': [0],
        'education_7th-8th': [0],
        'education_9th': [0],
        'education_Assoc-acdm': [0],
        'education_Assoc-voc': [0],
        'education_Bachelors': [0],
        'education_Doctorate': [0],
        'education_HS-grad': [0],
        'education_Masters': [0],
        'education_Preschool': [0],
        'education_Prof-school': [0],
        'education_Some-college': [0],
        'education-num_1': [0],
        'education-num_2': [0],
        'education-num_3': [0],
        'education-num_4': [0],
        'education-num_5': [0],
        'education-num_6': [0],
        'education-num_7': [0],
        'education-num_8': [0],
        'education-num_9': [0],
        'education-num_10': [0],
        'education-num_11': [0],
        'education-num_12': [0],
        'education-num_13': [0],
        'education-num_14': [0],
        'education-num_15': [0],
        'education-num_16': [0],
        'marital-status_Divorced': [0],
        'marital-status_Married-AF-spouse': [0],
        'marital-status_Married-civ-spouse': [0],
        'marital-status_Married-spouse-absent': [0],
        'marital-status_Never-married': [0],
        'marital-status_Separated': [0],
        'marital-status_Widowed': [0],
        'occupation_?': [0],
        'occupation_Adm-clerical': [0],
        'occupation_Armed-Forces': [0],
        'occupation_Craft-repair': [0],
        'occupation_Exec-managerial': [0],
        'occupation_Farming-fishing': [0],
        'occupation_Handlers-cleaners': [0],
        'occupation_Machine-op-inspct': [0],
        'occupation_Other-service': [0],
        'occupation_Priv-house-serv': [0],
        'occupation_Prof-specialty': [0],
        'occupation_Protective-serv': [0],
        'occupation_Sales': [0],
        'occupation_Tech-support': [0],
        'occupation_Transport-moving': [0],
        'relationship_Husband': [0],
        'relationship_Not-in-family': [0],
        'relationship_Other-relative': [0],
        'relationship_Own-child': [0],
        'relationship_Unmarried': [0],
        'relationship_Wife': [0],
        'race_Amer-Indian-Eskimo': [0],
        'race_Asian-Pac-Islander': [0],
        'race_Black': [0],
        'race_Other': [0],
        'race_White': [0],
        'sex_Female': [0],
        'sex_Male': [0],
        'native-country_?': [0],
        'native-country_Cambodia': [0],
        'native-country_Canada': [0],
        'native-country_China': [0],
        'native-country_Columbia': [0],
        'native-country_Cuba': [0],
        'native-country_Dominican-Republic': [0],
        'native-country_Ecuador': [0],
        'native-country_El-Salvador': [0],
        'native-country_England': [0],
        'native-country_France': [0],
        'native-country_Germany': [0],
        'native-country_Greece': [0],
        'native-country_Guatemala': [0],
        'native-country_Haiti': [0],
        'native-country_Holand-Netherlands': [0],
        'native-country_Honduras': [0],
        'native-country_Hong': [0],
        'native-country_Hungary': [0],
        'native-country_India': [0],
        'native-country_Iran': [0],
        'native-country_Ireland': [0],
        'native-country_Italy': [0],
        'native-country_Jamaica': [0],
        'native-country_Japan': [0],
        'native-country_Laos': [0],
        'native-country_Mexico': [0],
        'native-country_Nicaragua': [0],
        'native-country_Outlying-US(Guam-USVI-etc)': [0],
        'native-country_Peru': [0],
        'native-country_Philippines': [0],
        'native-country_Poland': [0],
        'native-country_Portugal': [0],
        'native-country_Puerto-Rico': [0],
        'native-country_Scotland': [0],
        'native-country_South': [0],
        'native-country_Taiwan': [0],
        'native-country_Thailand': [0],
        'native-country_Trinadad&Tobago': [0],
        'native-country_United-States': [0],
        'native-country_Vietnam': [0],
        'native-country_Yugoslavia': [0]
    }

    return pd.DataFrame(data=hot_encoded_data)


def predict_income(data):

    model_cache_key = 'model_cache'
    model_rel_path = "predictions/predictionmodel/model_cache/finalized_model.pkl"

    model = cache.get(model_cache_key)

    if not model:
        model_path = os.path.realpath(model_rel_path)
        model = joblib.load(model_path)
        # save in django memory cache
        cache.set(model_cache_key, model, None)

    donor_df = pd.DataFrame(data=data, index=[0])

    full_data = get_fullset_of_features()
    for feature in donor_df:
        for hot_encoded_feature in full_data:
            if str(feature) == str(hot_encoded_feature):
                full_data.at[0, str(hot_encoded_feature)] = donor_df.iloc[0][feature]
                continue
            elif str(hot_encoded_feature) == str(str(feature) + '_' + str(donor_df.iloc[0][feature])):
                full_data.at[0, str(hot_encoded_feature)] = 1
                continue

    pred = model.predict(full_data)

    if pred[0] == 0:
        return "<=50"
    else:
        return ">50"

