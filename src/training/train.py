# Import from third party
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import mixed_precision  # For faster GPU training
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Import from STL
import os


# TODOs:
# Test the model
# Print more metrics
# Retrain the model with less not moving data
# Increase dropout


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
        super().__init__()  # Enable Sequence optimizations (multithreading)

        # Pre-group DataFrame to avoid per-batch filtering
        self.groups: dict[int, pd.DataFrame] = {file_id: group for file_id, group in df.groupby("file_id")}

        self.df = df  # Keep ref on the dataframe
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Compute each start file offset
        self.file_starts = []
        self.sample_count = 0
        for file_id, group in self.groups.items():
            n_samples = max(0, len(group) - seq_len)  # Ensure no negative samples
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
        
        X_batch = np.zeros((self.batch_size, self.seq_len, len(self.feature_cols)), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, len(self.target_cols)), dtype=np.float32)

        # Loop through indices of the batch
        # Use pre-grouped dict instead of filtering
        # Fill batch with valid data only
        valid_count = 0
        remaining_indices = list(batch_idx)
        while valid_count < self.batch_size and remaining_indices:
            # Pop the next index to process
            idx = remaining_indices.pop(0)

            # Find the file and the corresponding position
            file_id, offset, sample_idx = self._get_file_and_idx(idx)
            group: pd.DataFrame = self.groups[file_id]
            X = group[self.feature_cols].values.astype(np.float32)
            y = group[self.target_cols].values.astype(np.float32)
            start = sample_idx
            if start + self.seq_len <= len(X):  # Check if enough data
                X_batch[valid_count] = X[start:start + self.seq_len]
                y_batch[valid_count] = y[start + self.seq_len]
                valid_count += 1
        
        # Handle case where no valid data is found
        if valid_count == 0:
            raise StopIteration  # Added to signal end of data, skips batch
        elif valid_count < self.batch_size:
            X_batch = X_batch[:valid_count]
            y_batch = y_batch[:valid_count]

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



def create_tf_dataset(generator: IMUSequenceGenerator) -> tf.data.Dataset:
    """
    Convert IMUSequenceGenerator to a tf.data.Dataset with prefetching.

    @param generator (IMUSequenceGenerator): The sequence generator instance
    @return tf.data.Dataset: Optimized dataset for training
    """
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, generator.seq_len, len(generator.feature_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(generator.target_cols)), dtype=tf.float32)
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)



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

    # Count rows with movement = 0 and movement = 1
    movement_0_count = len(df[df["movement"] == 0])
    movement_1_count = len(df[df["movement"] == 1])
    total_rows = len(df)
    print(f"\nMovement counts before downsampling:")
    print(f"Rows with movement = 0: {movement_0_count} ({movement_0_count/total_rows:.2%})")
    print(f"Rows with movement = 1: {movement_1_count} ({movement_1_count/total_rows:.2%})")

    # Downsample movement = 0 to 50% of its original count
    max_movement_0 = int(movement_0_count * 0.5)  # 50% of original movement = 0 rows
    df_movement_0: pd.DataFrame = df[df["movement"] == 0].sample(n=max_movement_0, random_state=42)  # Randomly select 50% (not moving so random is okay here)
    df_movement_1: pd.DataFrame = df[df["movement"] == 1]  # Keep all movement = 1 rows

    # Combine the downsampled movement = 0 with all movement = 1
    df = pd.concat([df_movement_0, df_movement_1], ignore_index=True)

    # Verify counts after downsampling
    movement_0_count_new = len(df[df["movement"] == 0])
    movement_1_count_new = len(df[df["movement"] == 1])
    total_rows_new = len(df)
    print(f"\nMovement counts after downsampling:")
    print(f"Rows with movement = 0: {movement_0_count_new} ({movement_0_count_new/total_rows_new:.2%})")
    print(f"Rows with movement = 1: {movement_1_count_new} ({movement_1_count_new/total_rows_new:.2%})")

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

    # Debug, train faster with only data of the first file
    #df = df[df["file_id"] == "01_undisturbed_slow_rotation_A.hdf5"]

    return df



if __name__ == "__main__":

    print("YOLO")
    print("TensorFlow version:", tf.__version__)
    print("TensorFlow build with cuda:", tf.test.is_built_with_cuda())
    print("Num GPUs available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices('GPU'))

    # Add GPU memory growth control
    gpus: list = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Prevent full VRAM pre-allocation
    mixed_precision.set_global_policy('mixed_float16')  # Enable FP16 for Tensor Cores

    # Load the data from the database
    print("\nLoading data from the database...")
    df: pd.DataFrame = loadFromDatabase("../dataExtraction/imu_data.db")#os.path.expanduser("~/DeepLearning_AHRS/src/dataExtraction/imu_data.db"))#

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

    # New: Split within each file
    train_dfs = []
    val_dfs = []
    test_dfs = []
    for file_id, group in df.groupby("file_id"):
        n_rows = len(group)
        train_size = int(n_rows * 0.7)  # 70% for train
        val_size = int(n_rows * 0.15)   # 15% for val
        test_size = n_rows - train_size - val_size  # Remainder (~15%) for test

        # Take sequential chunks to preserve temporal order
        train_df_part = group.iloc[:train_size].copy()
        val_df_part = group.iloc[train_size:train_size + val_size].copy()
        test_df_part = group.iloc[train_size + val_size:].copy()

        train_dfs.append(train_df_part)
        val_dfs.append(val_df_part)
        test_dfs.append(test_df_part)

    # Concatenate all parts
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    BATCH_SIZE: int = 64

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

    print(train_df.head())

    # Free unused DataFrames
    del df, train_df, val_df, test_df

    # tf.data.Dataset conversion
    train_dataset: tf.data.Dataset = create_tf_dataset(train_gen)
    val_dataset: tf.data.Dataset = create_tf_dataset(val_gen)
    test_dataset: tf.data.Dataset = create_tf_dataset(test_gen)

    # Test the generator
    #X_batch, y_batch = gen[0]
    #print(X_batch.shape, y_batch.shape)

    # Build the Model
    model: Sequential = Sequential([
        LSTM(128, input_shape=(100, 9), return_sequences=True),  # 100 timesteps, 9 features
        Dropout(0.3),  # Regularization to prevent overfitting
        LSTM(64),  # Second LSTM layer
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # Intermediate dense layer
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

    
    # Save the model if validation loss improves at each epoch
    checkpoint = ModelCheckpoint("checkpoint.h5", save_best_only=True, monitor='val_loss', mode='min')
    # Stop training if validation loss does not improve after 2 epoch. Then restore the best weights
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    # Train the Model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,  # Number of epochs
        callbacks=[checkpoint, early_stopping],  # Save the best model
        verbose=1  # Show training progress
    )

    # Evaluate on Test Set
    test_loss: float
    test_mae: float
    test_loss, test_mae = model.evaluate(test_dataset, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Save the model
    model.save("imu_lstm_model.h5")

    # Plot the training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.show()
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.legend()
    plt.show()