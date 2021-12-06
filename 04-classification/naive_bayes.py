import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


class NB_Gaussian():
    def __init__(self, training_df, test_df, p_class):
        self.training_df = training_df
        self.test_df = test_df
        self.p_class = p_class
        # training df grouped by classes, with calculated means and standard deviation
        self.aggs = self.training_df.groupby(self.p_class).agg(['mean', 'std'])

    # P(A|C) - for single example (with distinct feature from distinct group)
    # from gauss formula
    def cond_prob(self, row, feature, group):
        mu = self.aggs[feature]['mean'][group]  # mean
        std = self.aggs[feature]['std'][group]  # standard deviation

        # one selected observation (sigle value)
        selected_observ = self.test_df.loc[self.test_df['id'] == row][feature].values[0]

        prob = stats.norm.pdf(selected_observ, loc=mu, scale=std)

        return prob

    def predict_class(self, row):
        classes = []

        min_class = np.amin(self.training_df[self.p_class].values)
        max_class = np.amax(self.training_df[self.p_class].values)

        for group in range(min_class, max_class):
            # P(C)
            class_amount = len(self.training_df[self.training_df[self.p_class] == group])

            prob = class_amount/len(self.training_df.index)
            for feature in self.training_df.columns:
                if feature == self.p_class:
                    break
                # P(C) * P(A|C)
                prob *= self.cond_prob(row, feature, group)
            classes.append(prob)

        return np.argmax(classes) + min_class  # from what number classes starts

    def predict(self):
        values = dict()
        for index, row in self.test_df.iterrows():
            values[index] = self.predict_class(index)
        return values

    def check_result(self):
        res = {'True': 0, 'False': 0}
        predicted_val = self.predict()
        for index, row in self.test_df.iterrows():
            if self.test_df.at[index, self.p_class] == predicted_val[index]:
                res['True'] += 1
            else:
                res['False'] += 1
        return res


def split_set(df, ratio):
    train_df, test_df = train_test_split(df, train_size=ratio)
    return train_df, test_df


def cross_validation(df, k, p_class):
    shuffled = df.sample(frac=1)
    parts = np.array_split(shuffled, k)
    results = []

    for n in range(0, k):
        test_set = parts[n]
        rest = parts.copy()
        rest.pop(n)
        train_set = pd.concat(parts, ignore_index=True)
        gauss_classifier = NB_Gaussian(train_set, test_set, p_class)
        res = gauss_classifier.check_result()
        results.append(res)

    return results


def count_efficiency(results):
    good = 0
    all = 0
    for res in results:
        good += res['True']
        all += res['True'] + res['False']

    return f'well predicted: {good}\nfalsely predicted: {all-good}\nefficiency: {(good/all)*100}%'


if __name__ == "__main__":
    df = pd.read_csv('winequality-red.csv', sep=";")
    i = 0
    for index, row in df.iterrows():
        df.at[index, 'id'] = i
        i += 1
    p_class = 'quality'
    results = []

    print('Use cross validation? [y]/[n]')
    isCrossVal = input()
    if isCrossVal == 'y':
        print('put parameter k: ')
        k = int(input())
        results = cross_validation(df, k, p_class)
    else:
        print('put fraction of training set: ')
        ratio = float(input())
        if ratio >= 1:
            raise ValueError('Ratio should be less than 1')
        train_df, test_df = split_set(df, ratio)
        gauss_classifier = NB_Gaussian(train_df, test_df, p_class)
        results = [gauss_classifier.check_result()]

    print(results)
    print(count_efficiency(results))
