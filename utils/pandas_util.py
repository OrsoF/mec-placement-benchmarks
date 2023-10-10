import pandas as pd


def create_dataframe_from_dict(input_dict: dict):
    """
    Create a pandas.DataFrame from a dict {(row, col) : value}
    """
    # Extract unique row and column names
    rows = set()
    columns = set()
    for key in input_dict:
        row_name, column_name = key
        rows.add(row_name)
        columns.add(column_name)

    # Create an empty DataFrame with the extracted row and column names
    df = pd.DataFrame(index=rows, columns=columns)

    # Fill the DataFrame using the dictionary values
    for key, value in input_dict.items():
        row_name, column_name = key
        df.loc[row_name, column_name] = value

    return df


def reorder_columns_by_first_row(dataframe):
    # Get the values of the first row
    first_row_values = dataframe.iloc[0].values

    # Sort the indices of the columns based on the first row values
    sorted_indices = sorted(
        range(len(first_row_values)), key=lambda k: first_row_values[k]
    )

    # Reorder the columns in the DataFrame based on the sorted indices
    reordered_dataframe = dataframe.iloc[:, sorted_indices]

    return reordered_dataframe


def reorder_columns(df1, df2):
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        if df1.shape[0] >= 1:
            # Get the column order from the first row of df1
            column_order = df1.iloc[0].index.tolist()

            # Reorder columns in df1
            df1_reordered = df1[column_order]

            # Reorder columns in df2 based on the column order of df1
            df2_reordered = df2[column_order]

            return df1_reordered, df2_reordered
        else:
            raise ValueError("The first DataFrame must have at least one row.")
    else:
        raise TypeError("Input arguments must be Pandas DataFrames.")
