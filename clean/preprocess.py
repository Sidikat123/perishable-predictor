import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """"
      This function is used to encode the incoming data
    """

    nominal_columns = ['Product_Category', 'Region']  # one hot
    high_cardinal = ['Product_Name', 'Supplier_Name']  # mean encoding

    # OneHot Encoding for nominal columns
    encoded_columns = []
    ohe = OneHotEncoder(sparse_output=False, dtype=int)

    for col in nominal_columns:
        transformed_col = ohe.fit_transform(encode_data[[col]])
        column_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
        transformed_df = pd.DataFrame(transformed_col, columns=column_names, index=encode_data.index)
        encoded_columns.append(transformed_df)

        # Concatenate all encoded nominal columns
        encode_data = pd.concat([encode_data] + encoded_columns, axis=1)
        encode_data.drop(columns=nominal_columns, inplace=True)

    # Mean Encoding for high-cardinality columns using 'Units_Sold' as target
    for col in high_cardinal:
        encode_data[col] = encode_data.groupby(col)['Units_Sold'].transform('mean')