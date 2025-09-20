import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
import shutil
from time import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.base import clone
from unidecode import unidecode
import re
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)





def prepare(data,transpose=False):
    '''
    Prepare a DataFrame by combining column labels from the first few rows.

    This function takes a DataFrame as input and performs the following steps:
    1. If `transpose` is set to True, the DataFrame is transposed.
    2. Rows and columns containing only missing values are removed.
    3. The index of the DataFrame is reset.
    4. The resulting DataFrame is returned.

    Args:
        data (pandas.DataFrame): The input DataFrame to be prepared.
        transpose (bool, optional): If True, the DataFrame will be transposed before processing.
            Default is False.

    Returns:
        pandas.DataFrame: The prepared DataFrame with combined column labels.
    '''
    df=data.copy()
    if transpose==True:
        df=df.transpose()
    else:
        pass
    df.dropna(how='all',inplace=True)
    df.dropna(how='all',axis=1,inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def initiate_headers(data):
    '''
    Concatenate the first few rows of a DataFrame to create column headers.

    This function assumes that the first few rows of the DataFrame contain text that should be used as column headers.
    It concatenates these rows to create the column headers and replaces the original headers (which are assumed to be numbers).
    It is important to ensure that all the rows above the numeric data are text for this function to work correctly.
    If not all first rows are text, you can drop some rows and then reindex before applying this function.

    Args:
        data (pandas.DataFrame): The input DataFrame with headers named as numbers.

    Returns:
        pandas.DataFrame: The DataFrame with headers renamed using the concatenated text from the first few rows.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'0': ['Name', 'John', 'Alice'], '1': ['Age', 25, 30], '2': ['City', 'New York', 'London']})
        >>> df_with_headers = initiate_headers(df)
        >>> print(df_with_headers)
             name  age      city
        0    John   25  New York
        1   Alice   30    London

    Notes:
        - The function assumes that the middle column contains text in the first few rows.
        - It identifies the first row that does not contain text in the middle column and uses that as the starting point for numeric data.
        - The column names are concatenated using the text from the first few rows, separated by underscores.
        - The resulting column names are converted to lowercase and any hyphens are replaced with underscores.
        - The numeric data is retained and the DataFrame is reindexed starting from the first row of numeric data.
    '''
    df=data.copy()
    middle_column= df.columns[df.shape[1]//2]
    x=df.loc[~(df[middle_column].str.contains('[a-zA-Z]',na=False,regex=True)),middle_column].index[0]
    df[0:x-1]=df[0:x-1].values.astype(str)
    df.columns=df.iloc[0:x-1].fillna('').astype(str).apply(' '.join).str.strip()
    df.columns=df.columns.str.replace(" ","_")
    df.columns=df.columns.str.lower()
    df.columns=df.columns.str.replace("-","_")
    df=df.iloc[x:,:].reset_index(drop=True)
    return df    

def convert_string_to_nan(data):
    '''
    Convert string values containing English characters to NaN in a DataFrame.

    This function iterates over each column of the input DataFrame and replaces any string values
    that contain English characters (a-z or A-Z) with NaN (Not a Number) values. If a column
    cannot be processed (e.g., if it does not contain string values), a message is printed
    indicating that the column cannot be processed.

    Args:
        data (pandas.DataFrame): The input DataFrame containing columns with numbers and strings.

    Returns:
        pandas.DataFrame: The DataFrame with string values containing English characters converted to NaN.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 'abc', 4], 'B': [5, 'def', 7, 8], 'C': [9, 10, 11, 12]})
        >>> df_converted = convert_string_to_nan(df)
        >>> print(df_converted)
             A    B     C
        0  1.0  5.0   9.0
        1  2.0  NaN  10.0
        2  NaN  7.0  11.0
        3  4.0  8.0  12.0

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It uses the `str.contains()` method with a regular expression to identify string values containing English characters.
        - If a column cannot be processed (e.g., if it does not contain string values), a message is printed indicating the column name.
        - The function returns the modified DataFrame with string values containing English characters replaced with NaN.
    '''
    df=data.copy()
    for column in df.columns:
        try:
            df.loc[df[column].str.contains("[a-zA-Z]",na=False,regex=True),column]=np.nan
        except:
            print(f'{column} can not be processed')
    return df

def is_number(x):
    '''
    Check if a given value is a number.

    This function takes a value `x` and attempts to convert it to a float. If the conversion is successful,
    it means the value is a number, and the function returns True. If the conversion raises an exception,
    it means the value is not a number, and the function returns False.

    Args:
        x: The value to be checked.

    Returns:
        bool: True if the value is a number, False otherwise.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 'abc', 4], 'B': [5, 'def', 7, 8]})
        >>> df['A'].apply(is_number)
        0     True
        1     True
        2    False
        3     True
        Name: A, dtype: bool

    Notes:
        - The function uses a try-except block to attempt the conversion to a float.
        - If the conversion is successful (no exception is raised), the function returns True.
        - If the conversion raises an exception (e.g., ValueError for non-numeric strings), the function returns False.
        - This function can be used w
    '''
    try:
        float(x)
        return True
    except:
        return False

# to call nun numeric values in object column
def convert_to_number(data,number=np.nan):
    '''
    Convert non-numeric values in a DataFrame to a specified number.

    This function takes a DataFrame `data` and a value `number` (default is np.nan) as input. It converts all
    non-numeric values in the DataFrame to the specified `number`. The function creates a copy of the input
    DataFrame to avoid modifying the original data.

    Args:
        data (pandas.DataFrame): The input DataFrame containing numeric and non-numeric values.
        number (float, optional): The value to which non-numeric values will be converted. Default is np.nan.

    Returns:
        pandas.DataFrame: The DataFrame with non-numeric values converted to the specified number.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'A': [1, 2, 'abc', 4], 'B': [5, 'def', 7, 8]})
        >>> df_converted = convert_to_number(df, number=0)
        >>> print(df_converted)
           A  B
        0  1  5
        1  2  0
        2  0  7
        3  4  8

    Notes:
        - The function uses two helper functions: `is_not_number()` and `is_number()`.
        - `is_not_number()` checks if a value is not a number by attempting to convert it to a float. It returns True if the conversion raises an exception, indicating a non-numeric value.
        - `is_number()` checks if a value is a number by attempting to convert it to a float. It returns True if the conversion is successful, indicating a numeric value.
        - The function iterates over each column of the DataFrame and applies the `is_not_number()` function to each value using the `apply()` method.
        - Values for which `is_not_number()` returns True are replaced with the specified `number`.
        - The function returns the modified DataFrame with non-numeric values converted to the specified number.
    '''
    df=data.copy()
    
    def is_not_number(x):
        try:
            float(x)
            return False
        except:
            return True
        
    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False
    
    for column in df.columns:
        df.loc[df[column].apply(is_not_number),column]=number
    return df

def to_float(data,x=0):
    '''
    Convert specified columns of a DataFrame to float data type.

    This function takes a DataFrame `data` and an optional integer `x` (default is 0) as input. It converts
    the columns starting from the `x`-th column (0-indexed) to the end of the DataFrame to float data type,
    if possible. If a column cannot be converted to float (e.g., due to the presence of non-numeric values),
    it is skipped, and the conversion continues with the next column.

    Args:
        data (pandas.DataFrame): The input DataFrame containing columns to be converted to float.
        x (int, optional): The position of the first column to start converting from (0-indexed). Default is 0.

    Returns:
        pandas.DataFrame: The DataFrame with specified columns converted to float, if possible.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['4', '5', '6'], 'C': ['7.8', '9.0', '10.2']})
        >>> df_converted = to_float(df, x=1)
        >>> print(df_converted)
           A     B     C
        0  1   4.0   7.8
        1  2   5.0   9.0
        2  3   6.0  10.2

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It selects the columns starting from the `x`-th column to the end of the DataFrame using `df.columns[x:]`.
        - The function iterates over each selected column and attempts to convert it to float using `astype(float)`.
        - If a column cannot be converted to float (e.g., due to the presence of non-numeric values), a `ValueError` is raised, and the conversion for that column is skipped using a `try-except` block.
        - The function returns the modified DataFrame with the specified columns converted to float, if possible.
        - It is important to ensure that the columns to be converted do not contain `NaT` (Not a Time) values, as they prevent the conversion to float. Make sure to handle or remove `NaT` values before applying this function.
    '''
    df=data.copy()
    columns = df.columns[x:]
    for column in columns:
        try:
            df[column]=df[column].astype(float)
        except:
            pass
    return df

def to_numbers(data):
    '''
    Convert a DataFrame to numeric values and remove duplicated columns.

    This function takes a DataFrame `data` as input and converts its columns to numeric values. It first
    removes any duplicated columns by transposing the DataFrame, dropping duplicates, and then transposing
    it back. If there are still duplicated columns after this process, it removes them using boolean
    indexing. Finally, it converts each column to numeric values using `pd.to_numeric()` with the 'coerce'
    option, which replaces any non-numeric values with `NaN`.

    Args:
        data (pandas.DataFrame): The input DataFrame to be converted to numeric values.

    Returns:
        pandas.DataFrame: The DataFrame with duplicated columns removed and values converted to numeric.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['4', '5', '6'], 'C': ['7.8', '9.0', '10.2'], 'D': ['4', '5', '6']})
        >>> df_converted = to_numbers(df)
        >>> print(df_converted)
           A    B     C
        0  1  4.0   7.8
        1  2  5.0   9.0
        2  3  6.0  10.2

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It removes duplicated columns by transposing the DataFrame, dropping duplicates using `drop_duplicates()`, and then transposing it back.
        - If there are still duplicated columns after the previous step, it removes them using boolean indexing with `df.columns.duplicated()`.
        - The function then iterates over each column and converts it to numeric values using `pd.to_numeric()` with the 'coerce' option. This replaces any non-numeric values with `NaN`.
        - The function returns the modified DataFrame with duplicated columns removed and values converted to numeric.
    '''
    df2=data.copy()
    df = df2.T.drop_duplicates().T 
    if df.columns.duplicated().sum()>0:
        df = df.loc[:,~df.columns.duplicated()].copy()
    for column in df.columns:
        df[column]=pd.to_numeric(df[column],errors='coerce')
    return df

def df_outlier_columns(data,a=4, fill_na=True):
    '''
    Identify columns in a DataFrame that contain outliers based on a specified threshold.

    This function takes a DataFrame `data` and an optional parameter `a` (default is 4) as input and by default 
    it converts nan to zero via fill.na method.Itcalculates the z-scores of the numeric columns in the DataFrame
    using `stats.zscore()` from the `scipy.stats` module. The z-scores measure how many standard deviations each
    value is from the mean of its respective column. The function then identifies the columns that contain 
    any outliers, defined as values greater than `a` times the standard deviation of that column.

    Args:
        data (pandas.DataFrame): The input DataFrame to identify outlier columns.
        a (float, optional): The number of standard deviations used as the threshold for outliers. Default is 4.

    Returns:
        pandas.DataFrame: A DataFrame containing only the columns that have outliers greater than the specified threshold.

    Example:
        >>> import pandas as pd
        >>> from scipy import stats
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 1000], 'C': [2, 4, 6, 8, 10]})
        >>> outlier_cols = outlier_columns(df, a=3)
        >>> print(outlier_cols)
               B
        0    10
        1    20
        2    30
        3    40
        4  1000

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It calculates the z-scores of the numeric columns using `stats.zscore()` with `nan_policy='omit'` to handle missing values.
        - Missing values in the z-scores are filled with 0 using `fillna(0)`.
        - The absolute values of the z-scores are computed using `np.abs()`.
        - The function identifies the columns that contain any outliers greater than the specified threshold using boolean indexing.
        - The resulting DataFrame contains only the columns that have outliers.
        - The function assumes that the necessary libraries (`pandas`, `numpy`, and `scipy.stats`) are imported.
    '''
    df=data.copy()
    if fill_na:
        df.fillna(0,inplace=True)
    z_scores = stats.zscore(df[df.describe().columns],nan_policy='omit')
    z_scores.fillna(0,inplace=True)
    abs_z_scores = np.abs(z_scores)
    (abs_z_scores>a).any(axis=0)
    outliers_columns=abs_z_scores.columns[(abs_z_scores>a).any(axis=0)]
    return df[outliers_columns]

def df_without_outliers(data,a=4, fill_na=True ):
    '''
    Remove rows from a DataFrame that contain outliers based on a specified threshold.

    This function takes a DataFrame `data` and an optional parameter `a` (default is 4) as input. It
    calculates the z-scores of the numeric columns in the DataFrame using `stats.zscore()` from the
    `scipy.stats` module. The z-scores measure how many standard deviations each value is from the mean
    of its respective column. The function then identifies the rows that do not contain any outliers,
    defined as values greater than `a` times the standard deviation of their respective column. It
    returns a new DataFrame with the outlier rows removed.

    Args:
        data (pandas.DataFrame): The input DataFrame to remove outlier rows from.
        a (float, optional): The number of standard deviations used as the threshold for outliers. Default is 4.

    Returns:
        pandas.DataFrame: A new DataFrame with the outlier rows removed.

    Example:
        >>> import pandas as pd
        >>> from scipy import stats
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 1000], 'C': [2, 4, 6, 8, 10]})
        >>> df_no_outliers = df_without_outliers(df, a=3)
        >>> print(df_no_outliers)
           A   B   C
        0  1  10   2
        1  2  20   4
        2  3  30   6
        3  4  40   8

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It calculates the z-scores of the numeric columns using `stats.zscore()` with `nan_policy='omit'` to handle missing values.
        - Missing values in the z-scores are filled with 0 using `fillna(0)` to handle columns with all missing values.
        - The absolute values of the z-scores are computed using `np.abs()`.
        - The function identifies the rows that do not contain any outliers using boolean indexing with `(abs_z_scores < a).all(axis=1)`.
        - The resulting DataFrame `df_without_outliers` contains only the rows that do not have outliers.
        - The function assumes that the necessary libraries (`pandas`, `numpy`, and `scipy.stats`) are imported.
    '''
    df=data.copy()
    if fill_na:
        df.fillna(0,inplace=True)        
    z_scores = stats.zscore(df[df.describe().columns],nan_policy='omit')
    z_scores.fillna(0,inplace=True)   # in case one column is filled with nan values
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < a).all(axis=1)
    df_without_outliers = df[filtered_entries]
    return df_without_outliers

def cell_repetion(data,row_num=0):
    '''
    Rename cells in a specified row of a DataFrame based on the value of the previous cell in the same row.

    This function is typically used before initiating header names in a DataFrame. It takes a DataFrame `data`
    and an optional parameter `row_num` (default is 0) as input. The function iterates over each cell in the
    specified row and replaces the cell with the value of the previous cell in the same row if the current cell
    is not a string. This process repeats the values of cells until the end of the row is reached.

    Args:
        data (pandas.DataFrame): The input DataFrame to perform cell repetition on.
        row_num (int, optional): The row number (0-indexed) on which to perform cell repetition. Default is 0.

    Returns:
        pandas.DataFrame: A new DataFrame with the specified row modified by cell repetition.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': ['h2o', 1, 2, 3], 'B': [None, 4, 5, 6], 'C': ['hcl', 7, 8, 9]})
        >>> df_repeated = cell_repetition(df)
        >>> print(df_repeated)
             A    B    C
        0  h2o  h2o  hcl
        1    1    4    7
        2    2    5    8
        3    3    6    9

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It iterates over each column in the DataFrame using a for loop, starting from the second column.
        - For each cell in the specified row, it checks if the cell is not a string using `isinstance(df.loc[row_num, df.columns[i]], str)`.
        - If the cell is not a string, it assigns the value of the previous cell in the same row using `df.loc[row_num, df.columns[i]] = df.loc[row_num, df.columns[i-1]]`.
        - The function uses a try-except block to handle cases where the previous cell is out of bounds (i.e., the first cell in the row).
        - The modified DataFrame is returned as the output.
    '''
    df=data.copy()
    for i in df.columns:
        try:
            if type(df.loc[row_num,i+1])!=str:
                df.loc[row_num,i+1]=df.loc[row_num,i]
        except:
            pass
    return df



def cell_repetion_col(data, col_num=0):
    '''
    Rename cells in a specified column of a DataFrame based on the value of the previous non-empty cell in the same column.

    This function has the same functionality as 'cell_repetition', but instead of repeating cells row-wise, it repeats cells
    column-wise. It takes a DataFrame `data` and an optional parameter `col_num` (default is 0) as input. The function
    iterates over each cell in the specified column and replaces empty or NaN cells with the value of the previous
    non-empty cell in the same column. This process repeats the values of non-empty cells until the next non-empty cell
    is encountered.

    Args:
        data (pandas.DataFrame): The input DataFrame to perform cell repetition on.
        col_num (int, optional): The column number (0-indexed) on which to perform cell repetition. Default is 0.

    Returns:
        pandas.DataFrame: A new DataFrame with the specified column modified by cell repetition.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'A': ['h2o', np.nan, np.nan, np.nan], 'B': ['hcl', np.nan, np.nan, np.nan]})
        >>> df_repeated = cell_repetition_col(df, col_num=1)
        >>> print(df_repeated)
             A    B
        0  h2o  hcl
        1  NaN  hcl
        2  NaN  hcl
        3  NaN  hcl

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It iterates over each row in the DataFrame using a for loop.
        - For each cell in the specified column, it checks if the cell is empty or NaN using `pd.isnull(df.loc[i+1, col_num])`.
        - If the cell is empty or NaN, it assigns the value of the previous non-empty cell in the same column using `df.loc[i+1, col_num] = df.loc[i, col_num]`.
        - The function uses a try-except block to handle cases where the next cell is out of bounds (i.e., the last cell in the column).
        - The modified DataFrame is returned as the output.
    '''
    df = data.copy()
    for i in df.index:
        try:
            if pd.isnull(df.loc[i+1, col_num]):
                df.loc[i+1, col_num] = df.loc[i, col_num]
        except:
            pass
    return df 

def keep_columns(data,threshold=0.99):
    '''
    Remove columns from a DataFrame that contain a proportion of NaN values above a specified threshold.

    This function takes a DataFrame `data` and an optional parameter `threshold` (default is 0.5) as input.
    It removes columns from the DataFrame where the proportion of NaN values is greater than the specified threshold.
    In other words, it keeps columns that have a sufficient proportion of non-NaN values.

    Args:
        data (pandas.DataFrame): The input DataFrame to remove columns from.
        threshold (float, optional): The maximum proportion of NaN values allowed in a column.
            Default is 0.5 (i.e., remove columns with more than 50% NaN values).

    Returns:
        pandas.DataFrame: A new DataFrame with columns removed based on the specified threshold.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, np.nan, 7, 8], 'C': [9, 10, np.nan, np.nan]})
        >>> df_cleaned = keep_columns(df, threshold=0.5)
        >>> print(df_cleaned)
           A    B
        0  1  5.0
        1  2  NaN
        2  3  7.0
        3  4  8.0

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It uses the `dropna()` function from pandas to remove columns based on the specified threshold.
        - The `axis='columns'` parameter specifies that columns should be considered for removal.
        - The `thresh` parameter is calculated by multiplying the number of rows in the DataFrame (`df.shape[0]`)
          by `(1 - threshold)`. It represents the minimum number of non-NaN values required to keep a column.
        - The `inplace=True` parameter modifies the DataFrame in place.
        - The modified DataFrame is returned as the output.

    Additional Example:
        To remove columns that have 60% or more of their values as NaN, you can use:
        >>> df_cleaned = keep_columns(df, threshold=0.6)
        This will remove columns where the proportion of NaN values is greater than or equal to 60%.
    '''
    df=data.copy()
    df.dropna(axis='columns', how='any', thresh=int(df.shape[0]*threshold), inplace=True)
    return df

def keep_rows(data,threshold=0.99,indexreset=False):
    '''
    Remove rows from a DataFrame that contain a proportion of non-NA values below a specified threshold.

    This function takes a DataFrame `data`, an optional `threshold` value (default 0.99), and an optional
    `indexreset` flag (default False) as input. It removes rows from the DataFrame where the proportion of
    non-NA values is below the specified threshold. Additionally, it provides an option to reset the index
    of the resulting DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame to remove rows from.
        threshold (float, optional): The minimum proportion of non-NA values required to keep a row.
            Default is 0.99 (i.e., keep rows with at least 99% non-NA values).
        indexreset (bool, optional): Whether to reset the index of the resulting DataFrame. Default is False.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed based on the specified threshold and optional index reset.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'A': [1, 2, 3, np.nan], 'B': [4, 5, np.nan, np.nan], 'C': [7, 8, 9, np.nan]})
        >>> df_cleaned = keep_rows(df, threshold=0.6)
        >>> print(df_cleaned)
           A    B    C
        0  1.0  4.0  7.0
        1  2.0  5.0  8.0

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It uses the `dropna()` function from pandas to remove rows based on the specified threshold.
        - The `axis=0` parameter specifies that rows should be considered for removal.
        - The `how='any'` parameter specifies that a row should be removed if it contains any NA values.
        - The `thresh` parameter is calculated by multiplying the number of columns in the DataFrame (`df.shape[1]`)
          by the `threshold` value. It represents the minimum number of non-NA values required to keep a row.
        - The `inplace=True` parameter modifies the DataFrame in place.
        - If `indexreset` is set to True, the `reset_index()` function is used to reset the index of the resulting
          DataFrame, and the old index is dropped using `drop=True`.
        - The modified DataFrame is returned as the output.
    '''
    df=data.copy()
    df.dropna(axis=0, how='any', thresh=int(df.shape[1]*threshold), inplace=True)
    if indexreset==True:
        df.reset_index(drop=True,inplace=True)
    return df

def plotting(df,interactiv=False,title=True):
    '''
    Plot each column of a DataFrame against the index in separate plots.

    This function takes a DataFrame `df` and two optional parameters: `interactive` (default False) and `title` (default True).
    It plots each column of the DataFrame against the index in separate plots. If `interactive` is set to True, the charts
    will be interactive using the Plotly library. If `title` is set to True, the column name will be displayed as the title
    of each chart. If `title` is set to False, the column name will be used as the y-axis label.

    Args:
        df (pandas.DataFrame): The input DataFrame to plot.
        interactive (bool, optional): Whether to create interactive charts using Plotly. Default is False.
        title (bool, optional): Whether to display the column name as the title of each chart. Default is True.

    Returns:
        None

    Raises:
        None

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=[0, 1, 2])
        >>> plotting(df)
        # Plots each column of df against the index in separate plots with column names as titles

        >>> plotting(df, interactive=True, title=False)
        # Plots each column of df against the index in separate interactive plots with column names as y-axis labels

    Notes:
        - If `interactive` is set to False (default), the function uses Matplotlib to create static plots.
          - It iterates over each column in the DataFrame using a for loop.
          - For each column, it plots the column values against the DataFrame index using `df[column].plot()`.
          - If `title` is True (default), it sets the column name as the plot title using `plt.title(column)`.
          - If `title` is False, it sets the column name as the y-axis label using `plt.ylabel(column)`.
          - The x-axis label is set to 'hrs' using `plt.xlabel('hrs')`.
          - If a column cannot be plotted, it prints a message indicating that the column cannot be plotted.

        - If `interactive` is set to True, the function uses Plotly to create interactive plots.
          - It iterates over each column in the DataFrame using a for loop.
          - For each column, it creates a new Plotly figure using `fig = go.Figure()`.
          - It adds a scatter trace to the figure with the column values against the DataFrame index using `fig.add_trace()`.
          - If `title` is True (default), it sets the column name as the plot title using `fig.update_layout(title=column)`.
          - If `title` is False, it sets the column name as the y-axis label using `fig.update_layout(yaxis_title=column)`.
          - The x-axis label is set to 'Date' using `fig.update_layout(xaxis_title='Date')`.
          - If a column cannot be plotted, it prints a message indicating that the column cannot be plotted.

        - The function assumes that the necessary libraries (Matplotlib and Plotly) are imported and properly configured. 
    '''
    if interactiv==False:
        for column in df.columns:
            try:
                df[column].plot()
                plt.xlabel('hrs')                
                if title:
                    plt.title(column)
                else:
                    plt.ylabel(column)
                plt.show()
            except:
                print(f'{column} can not be plotted')
    else:
        for column in df.columns:
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[column],mode='lines+markers'))
                if title:
                    fig.update_layout(title= column, xaxis_title='Date')                   
                else:    
                    fig.update_layout(xaxis_title='Date', yaxis_title=column)
                fig.show()
            except:
                print(f'{column} can not be plotted')

def plotting_all(df,rescalling=False):
    '''
    Plot all columns of a DataFrame in the same interactive chart using Plotly.

    This function takes a DataFrame `df` and an optional parameter `rescaling` (default False).
    It plots all columns of the DataFrame in the same interactive chart using the Plotly library.
    If `rescaling` is set to True, all columns will be scaled from 0 to 100 using MinMaxScaler
    from the scikit-learn library.

    Args:
        df (pandas.DataFrame): The input DataFrame to plot.
        rescaling (bool, optional): Whether to scale all columns from 0 to 100. Default is False.

    Returns:
        None

    Raises:
        None

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=[0, 1, 2])
        >>> plotting_all(df)
        # Plots all columns of df in the same interactive chart using Plotly

        >>> plotting_all(df, rescaling=True)
        # Plots all columns of df in the same interactive chart with all columns scaled from 0 to 100

    Notes:
        - The function uses the Plotly library to create an interactive chart.
        - It creates a new Plotly figure using `fig = go.Figure()`.

        - If `rescaling` is set to True:
          - It applies MinMaxScaler from scikit-learn to scale all columns of the DataFrame from 0 to 100.
          - It creates a new DataFrame `df_transformed` with the scaled values.
          - It iterates over each column in `df_transformed` using a for loop.
          - For each column, it adds a scatter trace to the figure using `fig.add_trace()` with the scaled values.
          - The x-axis values are set to the DataFrame index using `x=df.index`.
          - The y-axis values are set to the scaled column values using `y=df_transformed[column]`.
          - The trace mode is set to 'lines+markers' to display both lines and markers.
          - The trace name is set to the column name using `name=column`.

        - If `rescaling` is set to False (default):
          - It iterates over each column in the original DataFrame `df` using a for loop.
          - For each column, it adds a scatter trace to the figure using `fig.add_trace()` with the original values.
          - The x-axis values are set to the DataFrame index using `x=df.index`.
          - The y-axis values are set to the column values using `y=df[column]`.
          - The trace mode is set to 'lines+markers' to display both lines and markers.
          - The trace name is set to the column name using `name=column`.

        - Finally, the function displays the interactive chart using `fig.show()`.

        - The function assumes that the necessary libraries (Plotly and scikit-learn) are imported and properly configured.
    '''
    if rescalling==True:
        fig = go.Figure()
        scaler = MinMaxScaler((0,100))
        scaler.fit(df[df.columns])
        df_transformed=pd.DataFrame(scaler.transform(df[df.columns]),columns=df.columns)
        for column in df_transformed.columns:
            fig.add_trace(go.Scattergl(x=df.index, y=df_transformed[column],mode='lines+markers', name=column))
        fig.show()
    else:
        fig = go.Figure()
        for column in df.columns:
            fig.add_trace(go.Scattergl(x=df.index, y=df[column],mode='lines+markers', name=column))
        fig.show()
                
def remove_spot(data):
    '''
   Remove the spot reading column from a raw DataFrame before transposing ( this is special for technical macro data in midor refinery).

    This function takes a DataFrame `data` as input and removes the spot reading column
    from it. The spot reading column is identified based on the following criteria:
    - The value in the 10th row of the column is not a string.
    - The value in the 10th row of the next column is a string.
    - The value in the 10th row of the previous column is a string.

    Args:
        data (pandas.DataFrame): The input DataFrame to remove the spot reading column from.

    Returns:
        pandas.DataFrame: A new DataFrame with the spot reading column removed.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': ['a', 'b', 'c'], 'Spot': [1, 2, 3], 'B': ['x', 'y', 'z']})
        >>> df_cleaned = remove_spot(df)
        >>> print(df_cleaned)
           A  B
        0  a  x
        1  b  y
        2  c  z

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - It iterates over each column in the DataFrame using a for loop.
        - For each column, it checks the following conditions:
          - The value in the 10th row of the column is not a string.
          - The value in the 10th row of the next column is a string.
          - The value in the 10th row of the previous column is a string.
        - If all the above conditions are satisfied, the column is considered as the spot reading column.
        - The spot reading column is dropped from the DataFrame using `df.drop(x, axis=1, inplace=True)`,
          where `x` is the column name.
        - If any of the conditions raise an exception (e.g., index out of bounds), the exception is caught
          and ignored using a try-except block.
        - The modified DataFrame with the spot reading column removed is returned as the output.

    Note:
        - The function assumes that the spot reading column is located between two columns with string
          values in the 10th row.
        - The function uses the `iloc` accessor to access values in the DataFrame by integer index.
        - The function uses `df.columns.get_loc(column)` to get the integer location of a column.
        - The function modifies the DataFrame in place using `inplace=True` in the `drop` method.
    '''
    df=data.copy()
    for column in df.columns:
        try:
            if (type(df.iloc[10,df.columns.get_loc(column)])!= str) and (type(df.iloc[10,df.columns.get_loc(column)+1])== str) and (type(df.iloc[10,df.columns.get_loc(column)-1])== str) :
                x=column
                df.drop(x,axis=1,inplace=True)
        except:
            pass
    return df

def remove_duplicated_columns(data, same_name_only=False):
    '''
    Remove duplicated columns from a DataFrame.

    This function takes a DataFrame `data` and removes duplicated columns based on the specified criteria.
    The `same_name_only` parameter determines whether duplicated columns are removed only if they have the
    same column name or even if they have different column names.

    Args:
        data (pandas.DataFrame): The input DataFrame to remove duplicated columns from.
        same_name_only (bool, optional): Flag to specify the criteria for removing duplicated columns.
            If True, duplicated columns will be removed only if they have the same column name.
            If False, duplicated columns will be removed even if they have different column names.
            Default is False.

    Returns:
        pandas.DataFrame: A new DataFrame with duplicated columns removed based on the specified criteria.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [1, 2, 3]})
        >>> df_cleaned = remove_duplicated_columns(df)
        >>> print(df_cleaned)
           A  B
        0  1  4
        1  2  5
        2  3  6

        >>> df_cleaned_same_name = remove_duplicated_columns(df, same_name_only=True)
        >>> print(df_cleaned_same_name)
           A  B  C
        0  1  4  1
        1  2  5  2
        2  3  6  3

    Notes:
        - The function creates a copy of the input DataFrame to avoid modifying the original data.
        - If `same_name_only` is set to False (default):
          - The function transposes the DataFrame using `df.T` to swap rows and columns.
          - It then drops duplicated rows (which were originally columns) using `drop_duplicates()`.
          - Finally, it transposes the DataFrame back to its original orientation using `.T`.
          - This process removes duplicated columns based on their values, regardless of their column names.
        - If `same_name_only` is set to True:
          - The function uses boolean indexing to select columns that do not have duplicated names.
          - It uses `df.columns.duplicated()` to identify duplicated column names.
          - The tilde (`~`) operator is used to invert the boolean mask, selecting columns that are not duplicated.
          - The resulting DataFrame only contains columns with unique names, removing any duplicates.

    Note:
        - The function assumes that the input DataFrame has a meaningful index and column names.
        - If `same_name_only` is False, the order of the columns in the resulting DataFrame may change
          compared to the original DataFrame.
        - If `same_name_only` is True, the order of the columns in the resulting DataFrame will be the
          same as the original DataFrame, but with duplicated columns removed.
    '''
    df=data.copy()
    if same_name_only == False:
        return df.T.drop_duplicates().T
    else:
        return df.loc[:,~df.columns.duplicated()]
       
def estimators_repeater(estimators=[RandomForestClassifier(),AdaBoostClassifier(),SVC()],tr_slicer=(None,None),tst_slicer=(None,None),loops=500,scorer=accuracy_score,X=None,y=None):
    '''
    """
    Train and evaluate a list of estimators on selected slices of a dataset for a specified number of iterations.

    This function trains a list of supplied estimators on selected slices of a dataset for a specified number of iterations (default 500).
    It calculates the training score, test score, and time used for each estimator in each iteration.

    Args:
        estimators (list): A list of estimator objects to be trained and evaluated. Default is [RandomForestClassifier(), AdaBoostClassifier(), SVC()].
        tr_slicer (tuple): A tuple of integers (starter, ender) specifying the slice of the training data to be used. Default is (None, None), which uses all samples.
        tst_slicer (tuple): A tuple of integers (starter, ender) specifying the slice of the test data to be used. Default is (None, None), which uses all samples.
        loops (int): The number of iterations to perform. Default is 500.
        scorer (function): The scoring function to be used for evaluation. Default is accuracy_score from sklearn.metrics.
        X (array-like): The feature matrix, either as a DataFrame or a 2-dimensional numpy array.
        y (array-like): The target vector, either as a DataFrame, Series, or a numpy array.

    Returns:
        None

    Global Variables:
        training_score_df (pandas.DataFrame): A DataFrame containing the training scores for each estimator in each iteration.
        testing_score_df (pandas.DataFrame): A DataFrame containing the test scores for each estimator in each iteration.
        timing_df (pandas.DataFrame): A DataFrame containing the time used for fitting and predicting each estimator in each iteration.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.svm import SVC
        >>> from sklearn.metrics import accuracy_score
        >>> estimators_repeater(estimators=[RandomForestClassifier(), SVC()], tr_slicer=(0, 200), loops=400, scorer=accuracy_score, X=X, y=y)

    Notes:
        - Make sure to import all the necessary estimators and scoring functions before calling this function.
        - If using a scoring function other than accuracy_score, modify the code accordingly.
        - The function assumes that the input data X and y are properly formatted and compatible with the estimators.
        - The function uses train_test_split from sklearn.model_selection to split the data into training and test sets in each iteration.
        - The function uses clone from sklearn.base to create a clean copy of each estimator in each iteration to avoid any side effects.
        - The function prints the time taken for each iteration.
        - The function stores the training scores, test scores, and timing information in global variables for further analysis.
    
    '''
    
    training_score={}
    testing_score={}
    timing={}
    for clf in estimators:
        clf_name = clf.__class__.__name__
        training_score[clf_name]=[]
        testing_score[clf_name]=[]
        timing[clf_name]=[]
       
    for i in range (loops):
        k1=time()
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=i)
        for clf in estimators:
            a=time()
            clf_name = clf.__class__.__name__
            clean_clf=clone(clf)
            clean_clf.fit(X_train[tr_slicer[0]:tr_slicer[1]],y_train[tr_slicer[0]:tr_slicer[1]])
            training_score[clf_name].append(scorer(y_train[tr_slicer[0]:tr_slicer[1]],clean_clf.predict(X_train[tr_slicer[0]:tr_slicer[1]])))
            testing_score[clf_name].append(scorer(y_test[tst_slicer[0]:tst_slicer[1]],clean_clf.predict(X_test[tst_slicer[0]:tst_slicer[1]])))
            b=time()
            timing[clf_name].append(b-a)
        k2=time()
        print(f'loop number {i} out of {loops} took {k2-k1} seconds')
    
    global training_score_df
    training_score_df=pd.DataFrame(training_score)
    global testing_score_df
    testing_score_df=pd.DataFrame(testing_score)
    global timing_df
    timing_df=pd.DataFrame(timing)

def solve(f, a, b, tol): 
    '''
    Approximate a root of a function using the bisection method.

    This function approximates a root, R, of the function f bounded by a and b
    to within a tolerance | f(m) | < tol, where m is the midpoint between a and b.
    It uses a recursive implementation of the bisection method.

    Args:
        f (function): The function for which the root is to be approximated.
        a (float): The lower bound of the interval containing the root.
        b (float): The upper bound of the interval containing the root.
        tol (float): The tolerance within which the root is to be approximated.

    Returns:
        float: The approximated root of the function.

    Raises:
        Exception: If the scalars a and b do not bound a root (i.e., f(a) and f(b) have the same sign).

    Example:
        >>> f = lambda x: x**2 - 2
        >>> r1 = solve(f, 0, 2, 0.1)
        >>> print("r1 =", r1)
        r1 = 1.4375
        >>> r01 = solve(f, 0, 2, 0.01)
        >>> print("r01 =", r01)
        r01 = 1.4140625
        >>> print("f(r1) =", f(r1))
        f(r1) = 0.06640625
        >>> print("f(r01) =", f(r01))
        f(r01) = -0.00042724609375

    Notes:
        - The function uses the bisection method to approximate the root.
        - It recursively divides the interval [a, b] into subintervals and selects the subinterval
          containing the root based on the signs of f(a), f(b), and f(m), where m is the midpoint.
        - The function stops when the absolute value of f(m) is less than the specified tolerance.
        - It is assumed that f is a continuous function and that a and b bound a root (i.e., f(a) and f(b) have opposite signs).
        - The function raises an exception if a and b do not bound a root.
     '''
    
    # check if a and b bound a root
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception(
         "The scalars a and b do not bound a root")
        
    # get midpoint
    m = (a + b)/2
    
    if np.abs(f(m)) < tol:
        # stopping condition, report m as root
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        # case where m is an improvement on a. 
        # Make recursive call with a = m
        return solve(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        # case where m is an improvement on b. 
        # Make recursive call with b = m
        return solve(f, a, m, tol)

def standerize_columns_names(data):
    """
    Standardize the column names of a DataFrame to snake_case pattern.

    This function takes a DataFrame as input, creates a copy of it, and renames its columns to follow the snake_case
    naming convention. It converts the column names to lowercase, removes leading/trailing whitespace, replaces spaces
    with underscores, removes any non-alphanumeric characters, and applies ASCII transliteration to convert non-ASCII
    characters to their closest ASCII equivalents. The modified copy of the DataFrame is then returned.

    Args:
        data (pandas.DataFrame): The DataFrame whose column names need to be standardized.

    Returns:
        pandas.DataFrame: A new DataFrame with standardized column names.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A B': [1, 2], 'C_D': [3, 4], 'E.F': [5, 6], 'Résumé': [7, 8]})
        >>> new_df = rename_columns(df)
        >>> print(new_df.columns)
        Index(['a_b', 'c_d', 'ef', 'resume'], dtype='object')

    Notes:
        - The function creates a copy of the input DataFrame using `data.copy()` to avoid modifying the original DataFrame.
        - It applies the following transformations to each column name:
            - Converts the name to lowercase using `str.lower()`.
            - Removes leading/trailing whitespace using `str.strip()`.
            - Replaces spaces with underscores using `str.replace(' ', '_')`.
            - Removes any non-alphanumeric characters using `re.sub('\W', '', ...)`.
            - Applies ASCII transliteration using `unidecode()` to convert non-ASCII characters to their closest ASCII equivalents.
        - The resulting column names follow the snake_case naming convention, where words are separated by underscores and only contain lowercase letters, digits, and underscores.
        - The function modifies the copied DataFrame in place using `df.rename()` with `inplace=True`.
        - The modified copy of the DataFrame is returned as the output.
        - It is assumed that the necessary libraries (`re` and `unidecode`) are imported before calling the function.
    """
    for col in df.columns:
        df=data.copy()
        new_col_name = re.sub('\W','',unidecode(col.lower().strip().replace(' ','_')))
        df.rename(columns = {col:new_col_name}, inplace = True)
        return df



def analyse_df(df, corr_limit = 0.75):
    """
    Analyze a DataFrame and print various statistical information.

    This function takes a DataFrame as input and performs the following analyses:
    - Prints general information about the DataFrame, including shape, duplicate rows, memory usage, and data types.
    - Calls DataFrame.describe() to display descriptive statistics.
    - Checks for missing values in each column and prints the quantity and percentage of missing values.
    - Checks for linear correlation between columns using Pearson correlation coefficient and prints the correlated columns.

    Args:
        df (pandas.DataFrame): The DataFrame to be analyzed.
        corr_limit (float, optional): The correlation limit (Pearson) to define if a relationship exists. Default is 0.75.

    Returns:
        None

    Example:
        >>> import pandas as pd
        >>> data = {'A': [1, 2, 3, 4, 5],
        ...         'B': [2.5, 3.7, 1.8, 4.2, 5.1],
        ...         'C': ['apple', 'banana', 'orange', 'apple', 'grape'],
        ...         'D': [True, False, True, False, True]}
        >>> df = pd.DataFrame(data)
        >>> analyse_df(df)
        General Info:
        5 Rows 4 Columns
        0 Duplicated Rows
        Memory Usage: 0.00Mb
        
        Columns int64: ['A']
        
        Columns float64: ['B']
        
        Columns object: ['C']
        
        Columns bool: ['D']
               A    B
        count  5.0  5.0
        mean   3.0  3.46
        std    1.58 1.35
        min    1.0  1.8
        25%    2.0  2.5
        50%    3.0  3.7
        75%    4.0  4.2
        max    5.0  5.1
        
        Cheking Missing Values:
        Analyzed DataFrame has no missing values
        
        Checking Linear Correlation:
        No linear correlation was found

    Notes:
        - The function uses various pandas methods and attributes to analyze the DataFrame.
        - It prints the general information about the DataFrame, including shape, duplicate rows, memory usage, and data types.
        - It categorizes the columns based on their data types (int64, float64, object, bool, other) and prints the column names for each category.
        - It calls DataFrame.describe() to display descriptive statistics of the numeric columns.
        - It checks for missing values in each column using DataFrame.isna().sum() and prints the quantity and percentage of missing values.
        - It calculates the linear correlation between columns using DataFrame.corr() and prints the correlated columns based on the specified correlation limit.
        - The function assumes that the input is a valid pandas DataFrame.

    """   

    print('General Info:')
    print(f'{df.shape[0]} Rows {df.shape[1]} Columns'
          f'\n{df.duplicated().sum()} Duplicated Rows'
          f'\nMemory Usage: {df.memory_usage().sum()/(1024*1024):.2f}Mb')
    
    # Checking Data Types
    int_list, float_list,object_list,bool_list,other_list =[[] for i in range(5)]
    for col in df.columns:
        if df[col].dtype == 'int64':
            int_list.append(col)
        elif df[col].dtype == 'float64':
            float_list.append(col)
        elif df[col].dtype == 'object':
            object_list.append(col)
        elif df[col].dtype == 'bool':
            bool_list.append(col)
        else:
            other_list.append(col)
            
    for type_list,data_type in zip([int_list, float_list,object_list,bool_list,other_list],
                                   ['int64','float64','object','bool','other']):
        if len(type_list)>0:
            print(f'\nColumns {data_type}: {type_list}')
            
    # General statistics
    display(df.describe())
    
    # Checking Missing Values in each columns
    print('\nCheking Missing Values:')
    col_with_missing_counter = 0
    for col in df.columns:
        qnt_missing = df[col].isna().sum()
        if qnt_missing > 0:
            col_with_missing_counter +=1
            print(f'Column "{col}" has {qnt_missing} missing values ({qnt_missing/df.shape[0]:.2%})')
    if col_with_missing_counter ==0 :
        print('Analyzed DataFrame has no missing values')
        
    # Checking linear correlation between columns
    print('\nChecking Linear Correlation:')
    df_corr = df.corr() # Correlation DataFrame
    ckecked_list =[] # Ensure that we won't print the same information twice
    cols_with_correlation_counter = 0
    for col in df_corr.columns:
        ckecked_list.append(col)
        for i in range(len(df_corr)):
            if ((df_corr[col][i] > corr_limit or df_corr[col][i] < -corr_limit) and
                (df_corr.index[i] not in ckecked_list)):
                cols_with_correlation_counter += 1
                print(f'Linear Correlation found between columns '
                      f'{df_corr.index[i]} and {col} -> Pearson coef. = {df_corr[col][i]:.2f}')         
    if cols_with_correlation_counter == 0:
        print('No linear correlation was found')

# Function to search for rows that contain specific terms
def search_rows(df, term1, term2=".*", term1_regex= False, term2_reg=True):
    """
    Search a pandas DataFrame for rows where at least one cell contains a specified term (`term1`)
    and another cell in the same row contains a second specified term (`term2`). Both terms can be
    searched as plain text or regular expressions.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to search through.
    - term1 (str): The first term to search for in the DataFrame. Can be a plain string or a regular expression based on the `term1_regex` flag.
    - term2 (str): The second term to search for in the DataFrame. Defaults to ".*" (matches anything). Can be a plain string or a regular expression based on the `term2_reg` flag.
    - term1_regex (bool): Flag to indicate whether `term1` should be treated as a regular expression (True) or plain text (False). Defaults to False.
    - term2_reg (bool): Flag to indicate whether `term2` should be treated as a regular expression (True) or plain text (False). Defaults to True.

    Returns:
    - pandas.Index: An Index object containing the indices of the rows that match the search criteria.

    
    Example:
  
        data = {
            'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Williams'],
            'Age': [25, 30, 35, 40],
            'City': ['New York', 'London', 'Paris', 'Tokyo']
        }
        df = pd.DataFrame(data)

        # Search for rows where 'Name' contains 'Doe' and 'City' contains 'New'
        matching_rows = search_rows(df, 'Doe', 'New')
        print(df.loc[matching_rows])

        Output:
        ```
            Name  Age      City
        0  John Doe   25  New York

        # Search for rows where 'Name' contains 'J' (as a regular expression)
        matching_rows = search_rows(df, 'J', term1_regex=True)
        print(df.loc[matching_rows])
        ```

        Output:
        ```
            Name  Age    City
        0  John Doe   25  New York
        1  Jane Smith   30  London
        2  Alice Johnson   35  Paris 

    Note:
    - The search is case-insensitive and ignores NaN values.
    - If `term2` is set to its default value of ".*", the function will return rows where at least one cell contains `term1`, regardless of the content of other cells.

    """
    # Checks if any cell contains term1 and another cell contains term2 (or anything if term2 is ".*")
    specific_rows = df[df.apply(lambda x: x.astype(str).str.contains(term1, regex=term1_regex, case=False,na=False)).any(axis=1) &
                        df.apply(lambda x: x.astype(str).str.contains(term2, regex=term2_reg,case =False, na=False)).any(axis=1)].index
    return specific_rows

def search_columns(df, term1, term2=".*",term1_reg= False, term2_reg=True):
    """
    Searches a pandas DataFrame for columns where at least one cell in the column contains both specified terms, `term1` and `term2`.
    Both terms can be searched as plain text or regular expressions within the same column's cells.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to search through.
    - term1 (str): The first term to search for within the cells of the DataFrame's columns. This can be a plain string or a regular expression, depending on the `term1_reg` flag.
    - term2 (str): The second term to search for within the same cells as `term1` in the DataFrame's columns. Defaults to ".*", which matches anything. This can also be a plain string or a regular expression, based on the `term2_reg` flag.
    - term1_reg (bool): Flag indicating whether `term1` should be treated as a regular expression (True) or plain text (False). Defaults to False.
    - term2_reg (bool): Flag indicating whether `term2` should be treated as a regular expression (True) or plain text (False). Defaults to True.

    Returns:
    - pandas.Index: An Index object containing the names of the columns that match the search criteria, where at least one cell in each column contains both `term1` and `term2`.

    Note:
    - The search is case-insensitive and ignores NaN values.
    - If `term2` is set to its default value of ".*", this effectively makes the presence of `term2` in the cell optional for the search criteria.

    Example:
    ```python
    import pandas as pd

    data = {
        'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Williams'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo'],
        'Email': ['john@example.com', 'jane@example.com', 'alice@example.com', 'bob@example.com']
    }
    df = pd.DataFrame(data)

    # Search for columns where at least one cell contains 'example' and 'com'
    matching_columns = search_columns(df, 'example', 'com')
    print(matching_columns)

    # Search for columns where at least one cell contains 'Doe' or 'Smith' (as a regular expression)
    matching_columns = search_columns(df, 'Doe|Smith', term1_reg=True)
    print(matching_columns)
    ```

    Output:
    ```
    Index(['Email'], dtype='object')
    Index(['Name'], dtype='object')
    ```
    """
    specific_columns = df.columns[df.apply(lambda x: x.astype(str).str.contains(term1, regex=term1_reg, case=False, na=False) &
                           x.astype(str).str.contains(term2, regex=term2_reg, case=False, na=False)).any()
    ]
    return specific_columns

        
