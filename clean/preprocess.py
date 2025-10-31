import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def feature_engineering(data: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Engineer features for training or inference.
    """
    data = data.copy()

    # Step 1: Storage Suitability
    data['Storage_Suitability'] = (
        data['Cold_Storage_Capacity'] * data['Shelf_Life_Days']
    )
    # Drop after engineering
    data.drop(columns=['Cold_Storage_Capacity', 'Shelf_Life_Days'], inplace=True)

    # Step 2: Wastage Rolling & Price Trend (only if training)
    if is_training:
        data = data.sort_values(by=['Product_ID', 'Store_ID', 'Week_Number'])
        
        data['Wastage_Rolling'] = (
            data.groupby(['Product_ID', 'Store_ID'])['Wastage_Units']
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=[0,1], drop=True)
        )
        
        data['Price_Trend'] = (
            data.groupby(['Product_ID', 'Store_ID'])['Price']
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=[0,1], drop=True)
        )
        
        # Drop raw columns
        data.drop(columns=['Wastage_Units', 'Price'], inplace=True, errors='ignore')

    # Step 3: Rainfall & Temperature Deviation
    if 'Rainfall' in data.columns and 'Avg_Temperature' in data.columns:
        data['Rainfall_Deviation'] = data['Rainfall'] - data.groupby('Region')['Rainfall'].transform('mean')
        data['Temperature_Deviation'] = data['Avg_Temperature'] - data.groupby('Region')['Avg_Temperature'].transform('mean')
        data.drop(columns=['Rainfall', 'Avg_Temperature'], inplace=True, errors='ignore')

    # Step 4: Marketing Intensity
    if 'Marketing_Spend' in data.columns:
        product_avg_marketing = data.groupby('Product_ID')['Marketing_Spend'].transform('mean')
        data['Marketing_Intensity'] = data['Marketing_Spend'] / (product_avg_marketing + 1)
        data.drop(columns=['Marketing_Spend'], inplace=True, errors='ignore')


    # Final cleanup
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    """"
      This function is used to Encode incoming data 
    """
  
    nominal_columns = ['Product_Category', 'Region']  # one hot
    high_cardinal = ['Product_Name']  # mean encoding

    # OneHot Encoding for nominal columns
    ohe = OneHotEncoder(sparse_output=False)

    encode_data = data.copy()
    encoded_columns = []

    # OneHot Encoding for nominal columns
    print("Columns before encoding:", encode_data.columns.tolist())  
    print("Data going into encoder:\n", encode_data[nominal_columns].head()) 
    
    for col in nominal_columns:
        transformed_col = ohe.fit_transform(encode_data[[col]])
        transformed_df = pd.DataFrame(transformed_col, columns = [f"{col}_{cat}" for cat in ohe.categories_[0]])
        
        encoded_columns.append(transformed_df)
        
    encode_data = pd.concat([encode_data] + encoded_columns, axis = 1)
    encode_data.drop(columns = nominal_columns, inplace = True)

    # Mean Encoding for high-cardinality columns using 'Price' as target
    for col in high_cardinal:
        mean_encode = encode_data.groupby(col)['Price'].mean()
        encode_data[col] = encode_data[col].map(mean_encode).fillna(0)
    
    encode_data.to_csv('encode_data.csv', index=False)
    return encode_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    '''Calls all the function above'''
    
    data = feature_engineering(data)
    data = encode_data(data)

    print("Cleaned data columns:", data.columns.tolist())
    
    return data
