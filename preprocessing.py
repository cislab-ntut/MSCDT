"""
Implement the preprocessing of nursery dataset
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.utils import shuffle


def nursery(path):
    names = ['parents', 'has_nurs', 'form', 'children',
             'housing', 'finance', 'social', 'health', 'label']
    return pd.read_csv(path, header=None, names=names, index_col=False)


def weather(path):
    return pd.read_csv(path, index_col=False)

def course(path):
    return pd.read_csv(path, index_col=False)


def heart(path):
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    return pd.read_csv(path, header=None, names=names, index_col=False)

def malware(path):
    return pd.read_csv(path, index_col=False)
def bank(path):
    return pd.read_csv(path, index_col=False)

def transform1(data):
    data.parents = data.parents.replace("usual", 1).replace(
        "pretentious", 2).replace("great_pret", 3).astype("int32")
    data.has_nurs = data.has_nurs.replace("proper", 1).replace("less_proper", 2).replace(
        "improper", 3).replace("critical", 4).replace("very_crit", 5).astype("int32")
    data.form = data.form.replace("complete", 1).replace("completed", 2).replace(
        "incomplete", 3).replace("foster", 4).astype("int32")
    data.children = data.children.replace("more", 4).astype("int32")
    data.housing = data.housing.replace("convenient", 1).replace(
        "less_conv", 2).replace("critical", 3).astype("int32")
    data.finance = data.finance.replace(
        "convenient", 1).replace("inconv", 2).astype("int32")
    data.social = data.social.replace("nonprob", 1).replace(
        "slightly_prob", 2).replace("problematic", 3).astype("int32")
    data.health = data.health.replace("recommended", 2).replace(
        "priority", 1).replace("not_recom", 3).astype("int32")
    data.label = data.label.replace("not_recom", 1).replace("recommend", 2).replace(
        "very_recom", 3).replace("priority", 4).replace("spec_prior", 5).astype("int32")
    return data


def transform2(data):
    data.drop(columns=["Date", "Evaporation", "Sunshine",
                       "Cloud3pm", "Cloud9am", "Location"], inplace=True)
    mms = MinMaxScaler()

    fill_column = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                   'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RISK_MM']
    data[fill_column] = data[fill_column].fillna(0)
    data[fill_column] = np.round(mms.fit_transform(
        data[fill_column])*5).astype("int32")

    data = pd.get_dummies(
        data, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'])
    data = data.dropna(subset=['RainToday', 'RainTomorrow'])
    data[['RainToday', 'RainTomorrow']] = data[['RainToday', 'RainTomorrow']
                                               ].replace("No", 1).replace("Yes", 2).astype("int32")
    data = data.rename(columns={"RainTomorrow": "label"})
    return data


def transform3(data):
    mms = MinMaxScaler()
    fill_column = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    data[fill_column] = data[fill_column].replace('?', 0)
    data[fill_column] = np.round(mms.fit_transform(
        data[fill_column])*10).astype("int32")
    data.label = data.label + 1
    return data

def transform4(data):
    mms = MinMaxScaler()
    fill_column = ['age', 'balance']
    data.job  = data.job.replace("admin.", 1).replace("blue-collar", 2).replace(
        "entrepreneur", 3).replace("housemaid", 4).replace("management", 5).replace("retired", 6).replace(
            "self-employed", 7).replace("services", 8).replace("student", 9).replace("technician", 10).replace("unemployed", 11).replace("unknown", 12).astype("int32")
    data.marital  = data.marital.replace("divorced", 1).replace("married", 2).replace(
    "single", 3).replace("unknown", 4).astype("int32")
    data.default = data.default.replace("no", 2).replace(
    "yes", 1).replace("unknown", 3).astype("int32")
    data.housing = data.housing.replace("no", 2).replace(
    "yes", 1).replace("unknown", 3).astype("int32")
    data.loan = data.loan.replace("no", 2).replace(
    "yes", 1).replace("unknown", 3).astype("int32")
    data.poutcome = data.poutcome.replace("failure", 2).replace(
    "success", 1).replace("unknown", 3).replace("other", 3).astype("int32")
    data.deposit = data.deposit.replace("yes", 1).replace("no", 2).astype("int32")
    data = data.rename(columns={"deposit": "label"})
    data.drop(columns=["contact", "day", "month","duration","education"], inplace=True)
    data[fill_column] = np.round(mms.fit_transform(
        data[fill_column])*5).astype("int32")
    
    return data
def transform5(data):
    fill_column = ['millisecond', 'policy','normal_prio','task_size','cached_hole_size','free_area_cache','mm_users','map_count','hiwater_rss','total_vm','reserved_vm',
    'nr_ptes','end_data','last_interval','nivcsw','min_flt','fs_excl_counter','lock','stime','gtime','cgtime','signal_nvcsw']
    mms = MinMaxScaler()
    data=data.dropna(how="any",axis=0)
    data['classification'] = data.classification.map({'benign':0, 'malware':1})
    data = data.rename(columns={"classification": "label"})
    data[fill_column] = np.round(mms.fit_transform(
        data[fill_column])*5).astype("int32")
    data=data.drop(["hash",'vm_truncate_count','shared_vm','exec_vm','nvcsw','maj_flt','utime'],axis=1)
    data[fill_column] = np.round(mms.fit_transform(
        data[fill_column])*5).astype("int32")
    return data

def transform6(data):
    fill_column = ['a', 'b','c','d','e']
    mms = MinMaxScaler()
    data = data.rename(columns={"f": "label"})
    data[fill_column] = np.round(mms.fit_transform(
        data[fill_column])*5).astype("int32")
    return data
def preprocessing(path):
    datas = {}
    names = ["heart", "nursery", "weather","bank","malware","學期成績"]
    func_ = [heart, nursery, weather,bank,malware,course]
    filter_ = [transform3, transform1, transform2,transform4,transform5,transform6]
    for i in range(0, 1):
        datas[names[i]] = shuffle(filter_[i](func_[i](path[i])))
    return datas
