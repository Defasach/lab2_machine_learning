#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

np.random.seed(42)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("Loading data...")
df = pd.read_csv('student/student-por.csv', sep=';')
print(f"Dataset shape: {df.shape}\n")

# Execute full pipeline
exec(open('lab2_sections.py').read())
