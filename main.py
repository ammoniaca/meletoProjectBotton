import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error, make_scorer


def _normalize(lst: list):
    value_max = lst[0]
    return [x / value_max for x in lst]


def _get_differences(lst: list):
    return [lst[i] - lst[i + 1] if lst[i + 1] != 0 else 0 for i in range(len(lst) - 1)]


def _multiply_by_weights(lst: list, weights):
    return [x * y for x, y in zip(lst, weights)]


def _dkl(lst: list):
    dkl = lst[0]
    if dkl == 1:
        return 0.5
    elif dkl == 0:
        return 0.0005
    else:
        return dkl


def _dls(lst: list):
    dls = sum(lst[1:])
    if dls <= 0:
        dls = 1
    return dls


def botton_score(f_lst: list, w=[6, 5, 4, 3, 2, 1]):
    # get number of non-zero F
    cf = sum(1 for x in f_lst if x != 0)
    inv_d = 1 / f_lst[0]
    # compute normalization of Fs
    f_norm = _normalize(f_lst)
    # compute differences form F_(i) and F_(i+1)
    f_diff = _get_differences(f_norm)
    # multiply by weights
    f_diff_w = _multiply_by_weights(lst=f_diff, weights=w)
    # compute dkl
    dkl = _dkl(f_diff_w)
    # compute dls
    dls = _dls(f_diff_w)
    ih = cf * dkl * dls * inv_d
    return ih if ih != 0 else 0.00064
    # return max(cf * dkl * dls * inv_d, 0.00064)


if __name__ == '__main__':
    # Esempio 1 test A. Botton
    # ex1 = [16.3, 15.9, 15.1, 14.8, 13.1, 0]
    # print(botton_score(ex1))
    df = pd.read_csv("RawDataset_All_varieties_All_treatments_02.05.24.csv", sep=";")
    df['target'] = df['Fruitdrop (%)'].str.replace('%', '').astype(float) / 100
    lst = df.iloc[:, 7:14].values.tolist()
    index = 0
    for l in lst:
        # print(index)
        print(botton_score(l))
        index = index + 1
        if index == 10:
            print('')