#!/usr/bin/env python3
"""Laboratory Work 2: Linear Regression - Complete Analysis"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, spearmanr, ttest_rel, wilcoxon
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import os

np.random.seed(42)
plt.rcParams['figure.dpi'] = 300
sns.set_style('whitegrid')

RESULTS = {'tables': {}, 'metrics': {}, 'hypotheses': {}}
REPORT = []

print("="*80)
print("LABORATORY WORK 2: LINEAR REGRESSION")
print("="*80)

# Load data
df = pd.read_csv('student/student-por.csv', sep=';')
print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target: G3, Missing values: {df.isnull().sum().sum()}")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
continuous_cols = [c for c in numerical_cols if c != 'G3']

print(f"Categorical: {len(categorical_cols)}, Numerical: {len(numerical_cols)}")

REPORT.append("# Laboratory Work №2: Linear Regression\n")
REPORT.append("## Student Performance Dataset Analysis\n")
REPORT.append(f"Dataset: {df.shape[0]} observations, {df.shape[1]} features\n")
REPORT.append(f"Target: G3 (final grade), Range: [{df['G3'].min()}, {df['G3'].max()}]\n")

