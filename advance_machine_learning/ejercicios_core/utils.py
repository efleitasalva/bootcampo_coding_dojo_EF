import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict, Any
import math
def cargar_datos(ruta_archivo):
    # Carga los datos del archivo CSV
    datos = pd.read_csv(ruta_archivo, delimiter=',')
    return datos

def exploracion_inicial(df):
    print("Informacion del dataset")
    print(df.info())
    print(df.head())
    ##Imprime ultimas 5 filas
    print("nÚltimas 5 filas del DataFrame:")
    print(df.tail(5))
    # Imprimir cantidad de filas y columnas del dataset
    print("Cantidad de filas: ", df.shape[0])
    print("Cantidad de columnas: ", df.shape[1])
# Identificar valores faltantes
def identificar_valores_faltantes(df):
    print("Valores faltantes en el dataset")
    qsna = df.shape[0] - df.isnull().sum(axis=0)
    qna = df.isnull().sum(axis=0)
    ppna = round(100 * (df.isnull().sum(axis=0) / df.shape[0]), 2)

    # Crear DataFrame con los resultados
    aux = {'datos sin NAs en q': qsna, 'Na en q': qna, 'Na en %': ppna}
    na_df = pd.DataFrame(data=aux)

    # Ordenar el DataFrame por el porcentaje de valores faltantes
    return na_df.sort_values(by='Na en %', ascending=False)
def verificacion_elementos_duplicados(df):
    ##Verificacion de elementos duplicados
    ###
    print("Cantidad de filas duplicadas: ",df.duplicated().sum())

### Verificar inconsistencias
#### Verificacion de inconsistencias en datos categoricos
def contar_valores_categoricos(df):
    # Recorre las columnas categóricas del DataFrame
    for col in df.select_dtypes(include=['object', 'category']):
        print(f'Valores únicos en la columna: {col}')
        print(df[col].value_counts())
        print('-' * 50)

##Se cambiar str to upper y strip utilizando metodo apply.
def modificar_valores_categoricos(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].str.upper().str.strip()
    return df
###Estadisiticas descriptivas
def estadisticas_descriptivas(df):
    print("Estadisticas del dataset")
    print(df.describe())    
    # Calcular medidas de tendencia central y dispersión para cada variable numérica
    print("Medidas de tendencia central y dispersión para cada variable numérica")
    measures = {}
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    ##Si los datos numericos estan en el dataframe
    if numeric_columns.size > 0:
        for column in numeric_columns:
            measures[column] = {
                'Mean': df[column].mean(),
                'Median': df[column].median(),
                'Mode': df[column].mode()[0],
                'Std Dev': df[column].std(),
                'Range': df[column].max() - df[column].min()
            }
    else:
        print("##No se encontraron columnas numéricas en el DataFrame.##")  

    # Mostrar medidas de tendencia central y dispersión
    for column, stats in measures.items():
        print(f"\n{column}:")
        for measure, value in stats.items():
            print(f"  {measure}: {value}")
    
### Verificacion de outliers
def plot_grafico_outliers(num_features,nrows, ncols, df):
    # Ajustar dinámicamente el tamaño de la figura
    figsize = (ncols * 5, nrows * 5)  # 5 unidades de ancho y alto por subplot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Aplanar para facilitar el acceso

    for i, feature in enumerate(num_features):
        sns.boxplot(x=df[feature], ax=axes[i], color="#75f8f2")
        axes[i].set_title(feature)

    # Ocultar ejes sobrantes si hay más subplots de los necesarios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

### Plot univariados de variables categoricas# Ajustar dinámicamente el tamaño de la figura
def plot_univariados_categoricos(df,ncols,nrows):
    figsize = (ncols * 5, nrows * 5)  # 5 unidades de ancho y alto por subplot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Aplanar para facilitar el acceso

    # Plotear variables categóricas
    for i, column in enumerate(df.select_dtypes(include=['object', 'category']).columns):
        sns.countplot(data=df, x=column, ax=axes[i])
        axes[i].set_title(column)

    # Ocultar ejes sobrantes si hay más subplots de los necesarios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def get_unique_values(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[Any]]:
    """
    Generates a dictionary where the keys are column names and the values are lists of unique values
    for the specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to process.
        columns (List[str]): A list of column names for which to retrieve unique values.

    Returns:
        Dict[str, List[Any]]: A dictionary with column names as keys and lists of unique values as values.

    Raises:
        ValueError: If a specified column is not found in the DataFrame.
    """
    unique_values_dict: Dict[str, List[Any]] = {}
    
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        unique_values_dict[column] = df[column].dropna().unique().tolist()
    
    return unique_values_dict
def save_dict_as_json(data: Dict[str, List[Any]], path: str, filename: str) -> None:
    """
    Saves a dictionary as a JSON file in the specified location.

    Args:
        data (Dict[str, List[Any]]): The dictionary to save.
        path (str): The directory where the file will be saved.
        filename (str): The name of the JSON file (including .json extension).

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    # Ensure the directory exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory '{path}' does not exist.")

    # Construct the full file path
    file_path = os.path.join(path, filename)
    
    # Save the dictionary as a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Dictionary saved to {file_path}")

def transform_columns(df: pd.DataFrame, columns: list, dtype: type) -> pd.DataFrame:
    """
    Generalized function to transform specified columns in a DataFrame to a given data type.
    
    Args:
    df (pd.DataFrame): The DataFrame to transform.
    columns (list): List of column names to transform.
    dtype (type or str): Desired data type ('category', 'int', 'float', 'bool', 'str', etc.).
    
    Returns:
    pd.DataFrame: The updated DataFrame with transformed columns.
    """
    for col in columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                    print(f"Error converting column '{col}' to {dtype}: {e}")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    return df
    