
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, spearmanr, normaltest
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from itertools import combinations

# Set seeds for reproducibility
np.random.seed(42)

# Global settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Results storage
results = {
    'tables': {},
    'plots': {},
    'metrics': {},
    'conclusions': {}
}

def load_data():
    """Load the student performance dataset"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv('student/student-por.csv', sep=';')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values: {df.isnull().sum().sum()} total")
    
    return df

def section_1_data_diagnosis(df):
    """Section 1: Data Diagnosis"""
    print("\n" + "=" * 80)
    print("SECTION 1: DATA DIAGNOSIS")
    print("=" * 80)
    
    # 1.1 Distribution Description
    print("\n1.1 Distribution Description")
    print("-" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    continuous_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Categorical features
    cat_stats = []
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        cat_stats.append({
            'feature': col,
            'unique_values': df[col].nunique(),
            'most_common': value_counts.index[0],
            'most_common_pct': f"{percentages.iloc[0]:.1f}%",
            'least_common': value_counts.index[-1],
            'least_common_pct': f"{percentages.iloc[-1]:.1f}%"
        })
    
    cat_df = pd.DataFrame(cat_stats)
    print("\nCategorical Features:")
    print(cat_df.to_string(index=False))
    results['tables']['categorical_distribution'] = cat_df
    
    # Continuous features
    cont_stats = []
    for col in continuous_cols:
        q25 = df[col].quantile(0.25)
        q50 = df[col].quantile(0.50)
        q75 = df[col].quantile(0.75)
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        cont_stats.append({
            'feature': col,
            'q25': f"{q25:.2f}",
            'q50': f"{q50:.2f}",
            'q75': f"{q75:.2f}",
            'skewness': f"{skew:.2f}",
            'kurtosis': f"{kurt:.2f}"
        })
    
    cont_df = pd.DataFrame(cont_stats)
    print("\nContinuous Features:")
    print(cont_df.to_string(index=False))
    results['tables']['continuous_distribution'] = cont_df
    
    # Target G3 statistics
    print("\nTarget G3 Statistics:")
    g3_stats = df['G3'].describe()
    print(g3_stats)
    results['tables']['g3_statistics'] = g3_stats
    
    # G3 histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['G3'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('G3 (Final Grade)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Variable G3')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('plots/g3_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved: plots/g3_distribution.png")
    
    # 1.2 Train/Val/Test Split
    print("\n1.2 Train/Val/Test Split (60/20/20)")
    print("-" * 80)
    
    # First split: 60% train, 40% temp
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    # Second split: 20% val, 20% test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\nData Leakage Explanation:")
    print("If we compute statistics (median, mean, boundaries) on the entire dataset")
    print("and then split, the model learns information about test data.")
    print("This inflates validation/test metrics and creates overly optimistic results.")
    print("Solution: Compute all transformations ONLY on train set, then apply to val/test.")
    
    return train_df, val_df, test_df, categorical_cols, continuous_cols
