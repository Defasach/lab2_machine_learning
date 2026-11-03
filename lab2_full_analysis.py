#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laboratory Work №2: Linear Regression - Complete Analysis
Student Performance Dataset
"""

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
plt.style.use('seaborn-v0_8-darkgrid')

# Storage for results
RESULTS = {'tables': {}, 'figures': {}, 'metrics': {}, 'hypotheses': {}}
REPORT_SECTIONS = []

print("="*80)
print("LABORATORY WORK №2: LINEAR REGRESSION")
print("Student Performance Dataset Analysis")
print("="*80 + "\n")

# Load data
print("Loading dataset...")
df = pd.read_csv('student/student-por.csv', sep=';')
print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Target variable: G3 (Final grade)")
print(f"✓ Missing values: {df.isnull().sum().sum()}\n")

# Identify column types
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
continuous_cols = [c for c in numerical_cols if c != 'G3']

print(f"Categorical features ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}...")
print(f"Numerical features ({len(numerical_cols)}): {', '.join(numerical_cols[:8])}...\n")
