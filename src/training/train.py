# Import from third party
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Import from STL



class IMUSequenceGenerator(Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        seq_len: int = 100,
        batch_size: int = 64,
        shuffle: bool = True
    ):
        """
        Generator initialization without storing in memory all the windows.
        """
        self.df = df  # Keep ref on the dataframe
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Compute each start file offset
        self.file_starts = []
        self.sample_count = 0
        for file_id, group in df.groupby("file_id"):
            n_samples = len(group) - seq_len
            if n_samples > 0:
                self.file_starts.append((file_id, self.sample_count, n_samples))
                self.sample_count += n_samples

        # Indices for the shuffeling. We shuffle the indices (the start of the sliding windows) and not the data.
        self.indices = np.arange(self.sample_count)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """
        Total number of batches.
        """
        return int(np.ceil(self.sample_count / self.batch_size))

    def __getitem__(self, index: int):
        """
        Generate one batch on the fly.
        """
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        X_batch = np.zeros((len(batch_idx), self.seq_len, len(self.feature_cols)), dtype=np.float32)
        y_batch = np.zeros((len(batch_idx), len(self.target_cols)), dtype=np.float32)

        # Loop through indices of the batch
        for i, idx in enumerate(batch_idx):
            # Find the file and the corresponding position
            file_id, offset, sample_idx = self._get_file_and_idx(idx)
            group = self.df[self.df["file_id"] == file_id]
            X = group[self.feature_cols].values.astype(np.float32)
            y = group[self.target_cols].values.astype(np.float32)
            
            # Extract the window
            start = sample_idx
            X_batch[i] = X[start:start + self.seq_len]
            y_batch[i] = y[start + self.seq_len]

        return X_batch, y_batch

    def _get_file_and_idx(self, idx):
        """Helper to get the file and the local index from a global index."""
        for file_id, start, n_samples in self.file_starts:
            if idx < start + n_samples:
                return file_id, start, idx - start
        raise IndexError("Index out of range")

    def on_epoch_end(self):
        """Blend indices at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)



def loadFromDatabase(database_path: str) -> pd.DataFrame:
    # Connect to the database
    conn = sqlite3.connect(database_path)

    # Load entire table into a DataFrame
    df: pd.DataFrame = pd.read_sql_query("SELECT * FROM imu_data", conn)

    # Close the connection
    conn.close()

    return df


def normalize_quaternions(df: pd.DataFrame, quat_cols: list[str]) -> pd.DataFrame:
    """
    Normalize quaternion columns to unit norm (per row).
    
    @param df (pd.DataFrame): DataFrame containing quaternion columns
    @param quat_cols (list[str]): List of column names ['quat_w', 'quat_x', 'quat_y', 'quat_z']
    
    @return pd.DataFrame: DataFrame with normalized quaternions
    """

    df = df.copy() # Ensures we're modifying a fresh copy, not a view

    q: np.ndarray = df[quat_cols].values
    norm: np.ndarray = np.linalg.norm(q, axis=1, keepdims=True)
    df[quat_cols] = q / norm
    return df


def preprocessData(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy() # Ensures we're modifying a fresh copy, not a view

    # Drop rows with NaN values in feature or target columns
    df = df.dropna()

    # Normalize the data
    # Centers signal around zero so LSTM learns more easily
    # Maps data to a standard normal distribution (mean=0, std=1), 
    # which can be better for capturing signed data (e.g., acceleration can be negative).
    # Values won’t be strictly between 0 and 1 (could be negative or >1), but LSTMs handle this fine

    scaler = StandardScaler()
    columns_to_scale = [
        "acc_x", "acc_y", "acc_z",
        "gyr_x", "gyr_y", "gyr_z",
        "mag_x", "mag_y", "mag_z"
    ]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Normalize the quaternions
    df = normalize_quaternions(df, ["quat_w", "quat_x", "quat_y", "quat_z"])

    # Drop the sample_idx columns
    df = df.drop(columns=["sample_idx"])

    # Drop the movement column
    df = df.drop(columns=["movement"])

    return df



if __name__ == "__main__":

    print("TensorFlow version:", tf.__version__)
    print("TensorFlow build with cuda:", tf.test.is_built_with_cuda())
    print("Num GPUs available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices('GPU'))

    # Load the data from the database
    print("\nLoading data from the database...")
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

    # Print the min and max values of each column of the DataFrame
    print("\nData min and max values:")
    print(df.describe().loc[["min", "max"]])

    # Preprocess the data
    print("\nPreprocessing data...")
    df = preprocessData(df)

    # Print the min and max values of each column of the DataFrame
    print("\nData min and max values:")
    print(df.describe().loc[["min", "max"]])

    feature_cols = ["acc_x", "acc_y", "acc_z",
                "gyr_x", "gyr_y", "gyr_z",
                "mag_x", "mag_y", "mag_z"]

    target_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]


    print("NaN in features:", df[feature_cols].isna().sum())
    print("NaN in targets:", df[target_cols].isna().sum())
    print("Inf in features:", np.isinf(df[feature_cols]).sum())
    print("Inf in targets:", np.isinf(df[target_cols]).sum())

    # Split by file_id to preserve temporal continuity
    file_ids: np.ndarray = df["file_id"].unique()

    # Split into train (70%), val (15%), test (15%)
    train_ids, temp_ids = train_test_split(file_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    # Create sub-DataFrames
    # For example, if file_ids = [1, 2, 3, 4, 5], train_ids might be [1, 3, 5] (70% of the IDs, randomly selected).
    train_df: pd.DataFrame = df[df["file_id"].isin(train_ids)].copy()
    val_df: pd.DataFrame = df[df["file_id"].isin(val_ids)].copy()
    test_df: pd.DataFrame = df[df["file_id"].isin(test_ids)].copy()

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    BATCH_SIZE: int = 256

    # Create generators
    train_gen: IMUSequenceGenerator = IMUSequenceGenerator(
        df=train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_len=100,
        batch_size=BATCH_SIZE,
        shuffle=True  # Shuffle windows within files for training
    )

    val_gen: IMUSequenceGenerator = IMUSequenceGenerator(
        df=val_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_len=100,
        batch_size=BATCH_SIZE,
        shuffle=False  # Preserve order for validation
    )

    test_gen: IMUSequenceGenerator = IMUSequenceGenerator(
        df=test_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_len=100,
        batch_size=BATCH_SIZE,
        shuffle=False  # Preserve order for testing
    )

    print(train_df.head(50))

    # Test the generator
    #X_batch, y_batch = gen[0]
    #print(X_batch.shape, y_batch.shape)

    # Build the Model
    model: Sequential = Sequential([
        LSTM(128, input_shape=(100, 9), return_sequences=True),  # 100 timesteps, 9 features
        Dropout(0.2),  # Regularization to prevent overfitting
        LSTM(64),  # Second LSTM layer
        Dropout(0.2),
        Dense(32, activation='relu'),  # Intermediate dense layer
        Dense(4, activation='linear')  # Output layer for 4 quaternion components
    ])

    # Compile the Model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error as additional metric
    )

    # Display model summary
    model.summary()
    #exit()

    # Train the Model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,  # Number of epochs
        verbose=1  # Show training progress
    )

    # Evaluate on Test Set
    test_loss: float
    test_mae: float
    test_loss, test_mae = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Save the model
    model.save("imu_lstm_model.h5")