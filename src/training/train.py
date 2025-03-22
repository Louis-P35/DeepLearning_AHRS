# Import from third party
import sqlite3
import numpy as np
import pandas as pd

# Import from STL



def loadFromDatabase(database_path: str) -> pd.DataFrame:
    # Connect to the database
    conn = sqlite3.connect(database_path)

    # Load entire table into a DataFrame
    df: pd.DataFrame = pd.read_sql_query("SELECT * FROM imu_data", conn)

    # Close the connection
    conn.close()

    return df



if __name__ == "__main__":

    df: pd.DataFrame = loadFromDatabase("../dataExtraction/imu_data.db")

    # Preview the data
    print("\nData preview:")
    print(df.head())

    # Print the shape of the DataFrame
    print("\nData shape:")
    print(df.shape)

    # Print the columns of the DataFrame
    print("\nData columns:")
    print(df.columns)

    # Print the data types of the DataFrame
    print("\nData types:")
    print(df.dtypes)

