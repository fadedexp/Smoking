import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import phik
from phik.report import plot_correlation_matrix
from phik import resources
from scipy import stats
from sklearn.model_selection import learning_curve, StratifiedKFold


class Feature_engineering():
    # feature_engineering
    def feature_engineering(self, df):
        df.loc[df['eyesight(left)'] > 3, 'eyesight(left)'] = 1
        df.loc[df['eyesight(right)'] > 3, 'eyesight(right)'] = 1
        df.loc[df['waist(cm)'] < 30, 'waist(cm)'] = df['waist(cm)'].median()
        
        df['BMI'] = df['weight(kg)'] / (df['height(cm)'] / 100) ** 2
        df['BP_Category'] = df.apply(self.bp_category, axis=1)
        df['Cholesterol_HDL_Ratio'] = df['Cholesterol'] / df['HDL']
        df['Liver_Function_Score'] = df['AST'] + df['ALT'] + df['Gtp']
        df['Kidney_Health_Indicator'] = df['serum creatinine'] / df['hemoglobin']
        df['Combined_Eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
        df['Combined_Hearing'] = (df['hearing(left)'] + df['hearing(right)']) / 2
        df['TG_HDL_Ratio'] = df['triglyceride'] / df['HDL']
        
        df = df.drop(columns=['eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)'])
        return df

    def features_to_drop(self, df, target='smoking', threshold=0.1):
        interval_cols = df.columns
        phik_target = df.phik_matrix(interval_cols=interval_cols)[target].to_frame().reset_index()
        features = phik_target[phik_target[target] <= threshold]['index']
        return features.to_list()


    def bp_category(self, row):
        if row['systolic'] < 120 and row['relaxation'] < 80:
            return 1
        elif 120 <= row['systolic'] < 130 and row['relaxation'] < 80:
            return 2
        elif 130 <= row['systolic'] < 140 or 80 <= row['relaxation'] < 90:
            return 3
        else:
            return 4


class Correlation_methods():
    def r_pb(self, binary_variable, continuous_variable):
        assert len(binary_variable) == len(continuous_variable)
        corr, p_value = stats.pointbiserialr(binary_variable, continuous_variable)
        return (corr, p_value)

    def cramers_v(self, binary_variable, catigorical_variable):
        assert len(binary_variable) == len(catigorical_variable)
        contingency_table = pd.crosstab(binary_variable, catigorical_variable)
        chi2, p, _, _ = stats.chi2_contingency(contingency_table)
        # Calculate Cramer's V
        n = contingency_table.sum().sum()  # Total number of observations
        k = min(df.shape)  # Minimum dimension (rows or columns)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))
        return cramers_v

    def phi_coefficient(self, binary_variable, binary_variable2):
        assert len(binary_variable) == len(binary_variable2)
        contingency_table = pd.crosstab(binary_variable, binary_variable2)
        a = contingency_table.loc[0,0]
        b = contingency_table.loc[0,1]
        c = contingency_table.loc[1,0]
        d = contingency_table.loc[1,1]
        phi = (a*d - b*c) / ((a+b)*(c+d)*(a+c)*(b+d))**0.5
        return phi

    def phi_coefficient_with_chi2(self, binary_variable, binary_variable2):
        assert len(binary_variable) == len(binary_variable2)
        contingency_table = pd.crosstab(binary_variable, binary_variable2)
        chi2, p, _, _ = stts.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        phi = np.sqrt(chi2 / n)
        return phi


class WoE():
    def WoE(self, df, feature_name, target_name):
        woe_df = df.groupby(feature_name)[target_name].value_counts().unstack().rename(columns={1: '# of events', 0:'# of non-events'})
        woe_df['Percentage events'] = woe_df['# of events'] / woe_df['# of events'].sum()
        woe_df['Percentage non-events'] = woe_df['# of non-events'] / woe_df['# of non-events'].sum()
        woe_df['WoE'] = np.log(woe_df['Percentage events'] / woe_df['Percentage non-events'])
        woe_df['IV'] = (woe_df['Percentage events'] - woe_df['Percentage non-events']) * woe_df['WoE'] 
        woe_df['Total Observations'] = woe_df['# of events'] + woe_df['# of non-events']
        woe_df['Percent of Observations'] = (woe_df['Total Observations'] / woe_df['Total Observations'].sum() * 100).round(3)
        return woe_df

    def plot_woe(self, woe, sort=False):
        if sort:
            woe = woe.sort_values(by='WoE', ascending=True)
        display(woe)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        sns.pointplot(data=woe, x=woe.index, y='WoE', color='red', linewidth=2, ax=ax1)
        ax1.set_ylabel('Weight of Evidence (WoE)', color='red')
        ax1.tick_params(axis='y', colors='red')
        ax1.set_xlabel('Cats')
        ax1.set_xticklabels(woe.index, rotation=90)
        sns.barplot(data=woe, x=woe.index, y='Percent of Observations', alpha=0.3, ax=ax2, errorbar=None)
        ax2.set_ylabel('Rate of Observations')
        ax2.set_ylim(0, woe['Percent of Observations'].max())
        ax2.axhline(y=5, color='red', linestyle='--', linewidth=1.5)
        ax1.grid()
        plt.show()

    def features_with_woe(self, df):
        df['age_cat'] = pd.cut(df['age'], bins=[0, 28, 39, 40, 50, 60, 86], labels=[6, 5, 4, 3, 2, 1])
        df['hemoglobin_cat'] = pd.cut(df['hemoglobin'], bins=[7, 12.3, 13, 13.5, 13.8, 14.3, 15.4, 16.4, 20], labels=list(range(1, 9)))
        df['height_cat'] = pd.cut(df['height(cm)'], bins=[130, 150, 155, 164, 200], labels=list(range(1, 5)))
        df['weight_cat'] = pd.qcut(df['weight(kg)'], q=7, labels=list(range(1, 8)))
        df['triglyceride_cat'] = pd.qcut(df['triglyceride'], q=8, labels=list(range(1, 9)))
        df['hdl_cat'] = pd.cut(df['HDL'], bins=[0, 40, 50, 57, 70, 75, 110], labels=list(range(6, 0, -1)))
        df['thr_cat'] = pd.cut(df['TG_HDL_Ratio'], bins=[0, .8, 0.95, 1.3, 1.5, 1.7, 2.1, 2.5, 3.4, 5, 13], labels=list(range(1, 11)))
        df['lfs_cat'] = pd.cut(df['Liver_Function_Score'], bins=[23, 42, 47, 51, 55, 62, 70, 80, 90, 110, 511], labels=list(range(1, 11)))
        df['chr_cat'] = pd.cut(df['Cholesterol_HDL_Ratio'], bins=[1, 2.5, 2.8, 3, 3.3, 3.6, 4, 4.3, 4.9, 8], labels=list(range(1, 10)))
        return df


class Plots():
    def heatmap_phik(self, df, target='smoking'):
        phik_target = df.phik_matrix()[target]
        display(df)
        plt.figure(figsize=(12, 8))
        sns.heatmap(phik_target.to_frame().sort_values(by=target, ascending=False), annot=True, cmap='Blues', cbar=True)
        plt.show()

    def plot_learning_curve(self, model, X, y, ylim=False):
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=skf, scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker='o')
        plt.plot(train_sizes, val_mean, label="Validation Score", color="orange", marker='o')

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="orange", alpha=0.2)
        if ylim: plt.ylim(0.75, 1)
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("AUC Score")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
