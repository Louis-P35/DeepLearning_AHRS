# Import from third party
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Lambda, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import mixed_precision  # For faster GPU training
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

# Import from STL
import os


# TODOs:
# Normalisation des Sorties : Force les quaternions prédits à être unitaires (norme 1)
    # model.add(Lambda(lambda x: x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))))

# Loss Fonction Adaptée
    # Problème : MSE traite toutes les composantes également, mais les erreurs angulaires comptent plus.
    # Solution : Ajoute une perte basée sur l’erreur angulaire

# Reduce model size

# predict euler angles

# Early stopping callback at 5 epochs without improvement

# Plus gros model & plus grand learning rate

# Pendant le training si ça saute d'un fichier à un autre ? ça va mettre du temps à reconverger et foiré le training.. ?


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Args:
        q (np.ndarray): Quaternion array of shape (4,) with [w, x, y, z].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def plot_cube(ax: Axes3D, position: np.ndarray, rotation_matrix: np.ndarray, color: str = 'b', label: str = None) -> None:
    """
    Plot a cube at a given position with a rotation matrix, including RGB axes at its center.

    Args:
        ax (Axes3D): Matplotlib 3D axis object.
        position (np.ndarray): 3D position of the cube center [x, y, z].
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
        color (str): Color of the cube edges (default: 'b').
        label (str, optional): Label to display near the cube (default: None).
    """
    # Vertices of a unit cube centered at origin
    r = [-0.5, 0.5]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r])
    # Apply rotation
    rotated_vertices = vertices @ rotation_matrix.T
    # Translate to position
    rotated_vertices += position

    # Cube edges (12 edges)
    edges = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ]
    
    # Plot cube edges
    for edge in edges:
        ax.plot3D(
            rotated_vertices[edge, 0],
            rotated_vertices[edge, 1],
            rotated_vertices[edge, 2],
            color=color
        )
    
    # Add RGB axes at the center of the cube
    axis_length = 0.6  # Length of the axes (slightly longer than cube edge for visibility)
    center = position  # Center of the cube
    
    # Define base axes (X, Y, Z) before rotation
    axes = np.array([
        [axis_length, 0, 0],  # X-axis (red)
        [0, axis_length, 0],  # Y-axis (green)
        [0, 0, axis_length]   # Z-axis (blue)
    ])
    
    # Apply rotation to the axes
    rotated_axes = axes @ rotation_matrix.T
    
    # Plot each axis from the center
    for i, axis_color in enumerate(['r', 'g', 'b']):
        ax.plot3D(
            [center[0], center[0] + rotated_axes[i, 0]],
            [center[1], center[1] + rotated_axes[i, 1]],
            [center[2], center[2] + rotated_axes[i, 2]],
            color=axis_color, linewidth=2
        )
    
    # Add label if provided
    if label:
        ax.text(position[0], position[1], position[2], label, color=color)

def update_frame(frame: int, ax: Axes3D, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Update the animation frame with two cubes for true and predicted quaternions.

    Args:
        frame (int): Current frame index.
        ax (Axes3D): Matplotlib 3D axis object.
        y_test (np.ndarray): Ground truth quaternions of shape (n_samples, 4).
        y_pred (np.ndarray): Predicted quaternions of shape (n_samples, 4).
    """
    ax.cla()  # Clear the axis
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Ground truth quaternion (left cube)
    q_true = y_test[frame]
    rot_true = quaternion_to_rotation_matrix(q_true)
    plot_cube(ax, position=[-1, 0, 0], rotation_matrix=rot_true, color='b', label='Truth')
    
    # Predicted quaternion (right cube)
    q_pred = y_pred[frame]
    rot_pred = quaternion_to_rotation_matrix(q_pred)
    plot_cube(ax, position=[1, 0, 0], rotation_matrix=rot_pred, color='r', label='Predicted')
    
    ax.set_title(f'Frame {frame}')

def generate_videos(y_test: np.ndarray, y_pred: np.ndarray, num_videos: int = 20, fps: int = 20, interval: int = 500) -> None:
    """
    Generate multiple videos showing cubes animated with ground truth and predicted quaternions.

    Args:
        y_test (np.ndarray): Ground truth quaternions of shape (n_samples, 4).
        y_pred (np.ndarray): Predicted quaternions of shape (n_samples, 4).
        num_videos (int): Number of videos to generate (default: 20).
        fps (int): Frames per second for the video (default: 20).
        interval (int): Delay between frames in milliseconds (default: 500).
    """
    print("Generating videos...")
    total_samples = len(y_test)
    samples_per_video = total_samples // num_videos  # Number of frames per video

    for i in range(num_videos):
        start_idx = i * samples_per_video
        end_idx = (i + 1) * samples_per_video if i < num_videos - 1 else total_samples
        n_frames = end_idx - start_idx

        # Subset of data for this video
        y_test_subset = y_test[start_idx:end_idx]
        y_pred_subset = y_pred[start_idx:end_idx]

        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create animation
        ani = FuncAnimation(
            fig, update_frame, frames=n_frames,
            fargs=(ax, y_test_subset, y_pred_subset), interval=interval
        )

        # Save video
        video_filename = f'quaternion_animation_part_{i+1:02d}.mp4'
        ani.save(video_filename, writer='ffmpeg', fps=fps)
        plt.close(fig)  # Close figure to free memory
        print(f"Video {i+1}/{num_videos} saved as '{video_filename}'")


class ResetStatesCallback(Callback):
    """
    Callback to reset LSTM states at each epoch start.
    """
    def __init__(self, lstm_layers):
        super(ResetStatesCallback, self).__init__()
        self.lstm_layers = lstm_layers

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.lstm_layers:
            layer.reset_states()
            print(f"Layer {layer.name} Reset")


def create_sequences(df, feature_cols, target_cols, seq_len=2):
    """
    Create sequences from a DataFrame, preloaded in memory.

    @param df (pd.DataFrame): The DataFrame containing the data
    @param feature_cols (list[str]): The list of feature columns
    @param target_cols (list[str]): The list of target columns
    @param seq_len (int): The sequence length
    @return np.ndarray, np.ndarray: The input and target
    """
    X, y = [], []
    for _, group in df.groupby("file_id"):
        X_group = group[feature_cols].values.astype(np.float32)
        y_group = group[target_cols].values.astype(np.float32)
        for i in range(len(X_group) - seq_len + 1):
            X.append(X_group[i:i + seq_len])
            y.append(y_group[i + seq_len - 1])
    return np.array(X), np.array(y)


def create_sequences_stateful(df, feature_cols, target_cols, seq_len, batch_size):
    X, y = [], []
    for _, group in df.groupby("file_id"):
        X_group = group[feature_cols].values.astype(np.float32)
        y_group = group[target_cols].values.astype(np.float32)
        n_samples = len(X_group)
        # Découper en blocs consécutifs sans chevauchement
        n_steps = (n_samples - seq_len + 1) // (batch_size * seq_len)
        for batch_idx in range(n_steps):
            batch_X = []
            batch_y = []
            start_idx = batch_idx * batch_size * seq_len
            for i in range(batch_size):
                seq_start = start_idx + i * seq_len
                seq_end = seq_start + seq_len
                if seq_end <= n_samples:
                    batch_X.append(X_group[seq_start:seq_end])
                    batch_y.append(y_group[seq_end - 1])
                else:
                    # Padding si fin du fichier
                    pad_len = seq_end - n_samples
                    X_pad = np.pad(X_group[seq_start:], ((0, pad_len), (0, 0)), mode='constant')
                    batch_X.append(X_pad)
                    batch_y.append(y_group[-1])  # Dernière valeur comme placeholder
            X.append(np.array(batch_X))
            y.append(np.array(batch_y))
    return np.concatenate(X), np.concatenate(y)


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


def quaternion_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the mean angular error loss between predicted and true quaternions.
    This loss measures the angular difference (in radians) between two unit quaternions.
    Although quaternions are normalized during preprocessing, we re-normalize to ensure stability.

    Parameters:
        y_true (tf.Tensor): Ground truth quaternions.
        y_pred (tf.Tensor): Predicted quaternions.

    Returns:
        tf.Tensor: Scalar tensor representing the mean angular error.
    """
    # Re-normalize to ensure unit norm (safety measure)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    
    # Compute the dot product between the quaternions
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    
    # Clip the dot product to the range [-1, 1] to avoid numerical errors with acos.
    dot = tf.clip_by_value(dot, -1.0, 1.0)
    
    # Compute the angular error (in radians); absolute value accounts for double-cover property.
    angle_error = tf.acos(tf.abs(dot))
    
    # Return the mean angular error over the batch.
    return tf.reduce_mean(angle_error)


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

    # Drop not moving samples
    df = df[df["movement"] != 0]

    # Normalize the quaternions
    df = normalize_quaternions(df, ["quat_w", "quat_x", "quat_y", "quat_z"])

    # Drop the sample_idx columns
    df = df.drop(columns=["sample_idx"])

    # Drop the movement column
    df = df.drop(columns=["movement"])

    # Debug, train faster with only data of the first file
    #df = df[df["file_id"] == "01_undisturbed_slow_rotation_A.hdf5"]

    return df


def generate_synthetic_data(n_samples=10000, n_files=5):
    """
    Generate a synthetic dataset with sine waves mimicking IMU and quaternion data.
    - 9 input features (like acc_x, acc_y, etc.) as sine waves with different frequencies.
    - 4 target columns (quat_w, quat_x, quat_y, quat_z) as phase-shifted sine waves.
    - Split into multiple "files" to mimic file_id grouping.
    """
    t = np.linspace(0, n_samples / 100, n_samples)  # Time vector
    data = []
    
    samples_per_file = n_samples // n_files
    for file_idx in range(n_files):
        start_idx = file_idx * samples_per_file
        end_idx = (file_idx + 1) * samples_per_file if file_idx < n_files - 1 else n_samples
        
        # Generate 9 input features (sine waves with different frequencies)
        features = {
            "acc_x": np.sin(1 * t[start_idx:end_idx]),
            "acc_y": np.sin(2 * t[start_idx:end_idx]),
            "acc_z": np.sin(3 * t[start_idx:end_idx]),
            "gyr_x": np.sin(4 * t[start_idx:end_idx]),
            "gyr_y": np.sin(5 * t[start_idx:end_idx]),
            "gyr_z": np.sin(6 * t[start_idx:end_idx]),
            "mag_x": np.sin(7 * t[start_idx:end_idx]),
            "mag_y": np.sin(8 * t[start_idx:end_idx]),
            "mag_z": np.sin(9 * t[start_idx:end_idx]),
        }

        print (start_idx, " ", end_idx)
        
        # Generate 4 target "quaternions" (simplified sine waves, not true quaternions)
        targets = {
            "quat_w": np.sin(1 * t[start_idx:end_idx] + 0.5),  # Phase shift
            "quat_x": np.cos(2 * t[start_idx:end_idx]),
            "quat_y": np.sin(3 * t[start_idx:end_idx] + 1.0),
            "quat_z": np.sin(4 * t[start_idx:end_idx]),
        }
        
        # Add file_id and dummy columns to match your structure
        df_file = pd.DataFrame({
            "file_id": [f"file_{file_idx:02d}"] * (end_idx - start_idx),
            "sample_idx": np.arange(end_idx - start_idx),
            "movement": ["synthetic"] * (end_idx - start_idx),
            **features,
            **targets
        })
        data.append(df_file)
    
    return pd.concat(data, ignore_index=True)


# Add this after preprocessing the data
def plot_synthetic_data(df, output_dir="synthetic_plots"):
    """
    Plot the synthetic data to visualize input features and targets.
    
    Args:
        df (pd.DataFrame): The synthetic DataFrame after preprocessing.
        output_dir (str): Directory to save the plots.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Select one file for plotting (e.g., the first file_id)
    file_id = df["file_id"].iloc[0]
    df_file = df[df["file_id"] == file_id].reset_index(drop=True)
    time = np.arange(len(df_file))  # Time index

    # Define feature and target columns
    feature_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z"]
    target_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]

    # 1. Time Series Plot for All Columns in One Graph
    plt.figure(figsize=(12, 6))
    for col in feature_cols:
        plt.plot(time, df_file[col], label=col, alpha=0.5)
    for col in target_cols:
        plt.plot(time, df_file[col], label=col, linewidth=2)
    plt.title(f"Synthetic Data - File: {file_id}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"time_series_{file_id}.png"))
    plt.close()

    # 2. Subplot Grid for Each Column
    n_cols = 13  # 9 features + 4 targets
    n_rows = 5   # Adjust as needed
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols + target_cols, 1):
        plt.subplot(n_rows, (n_cols + n_rows - 1) // n_rows, i)
        plt.plot(time, df_file[col], label=col)
        plt.title(col)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"subplot_grid_{file_id}.png"))
    plt.close()

    print(f"Plots saved in '{output_dir}' directory.")



if __name__ == "__main__":

    # Hyperparameters
    BATCH_SIZE: int = 256
    REGULARIZER: float = 0.01
    DROPOUT: float = 0.2
    LEARNING_RATE: float = 0.0005
    EPOCH: int = 100
    SEQUENCE_LENGTH: int = 20 # Window of x samples
    MODEL_SIZE: int = 512

    USE_SYNTHETHIC_DATA: bool = False

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
    df: pd.DataFrame = pd.DataFrame()
    if (not USE_SYNTHETHIC_DATA):
        df = loadFromDatabase("../dataExtraction/imu_data.db")#os.path.expanduser("~/DeepLearning_AHRS/src/dataExtraction/imu_data.db"))#
    else:
        df = generate_synthetic_data(n_samples=1000000, n_files=1)

    # Insert this after preprocessing in your main block
    #print("\nPlotting synthetic data...")
    #plot_synthetic_data(df)
    #exit()

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
    if (not USE_SYNTHETHIC_DATA):
        df = preprocessData(df)

    # Print the min and max values of each column of the DataFrame
    print("\nData min and max values:")
    print(df.describe().loc[["min", "max"]])

    feature_cols: list[str] = ["acc_x", "acc_y", "acc_z",
                "gyr_x", "gyr_y", "gyr_z",
                "mag_x", "mag_y", "mag_z"]

    target_cols: list[str] = ["quat_w", "quat_x", "quat_y", "quat_z"]


    print("NaN in features:", df[feature_cols].isna().sum())
    print("NaN in targets:", df[target_cols].isna().sum())
    print("Inf in features:", np.isinf(df[feature_cols]).sum())
    print("Inf in targets:", np.isinf(df[target_cols]).sum())

    # Split by file_id to preserve temporal continuity
    file_ids: np.ndarray = df["file_id"].unique()

    # Split within each file
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

    # Preload the sequences
    print("\nCreating sequences...")
    X_train, y_train = create_sequences_stateful(train_df, feature_cols, target_cols, SEQUENCE_LENGTH, BATCH_SIZE)
    X_val, y_val = create_sequences_stateful(val_df, feature_cols, target_cols, SEQUENCE_LENGTH, BATCH_SIZE)
    X_test, y_test = create_sequences_stateful(test_df, feature_cols, target_cols, SEQUENCE_LENGTH, BATCH_SIZE)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    print("y_train range:", y_train.min(), y_train.max())
    print("y_test range:", y_test.min(), y_test.max())

    # Convert in tf.data.Dataset for batching efficiency
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # Build the Model
    inputs = Input(batch_shape=(BATCH_SIZE, SEQUENCE_LENGTH, 9))  # Batch size fixe
    x = LSTM(int(MODEL_SIZE), stateful=True, return_sequences=True, kernel_regularizer=l2(REGULARIZER), name="lstm_1")(inputs)
    x = Dropout(DROPOUT)(x)
    x = LSTM(int(MODEL_SIZE/2), stateful=True, kernel_regularizer=l2(REGULARIZER), name="lstm_2")(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(int(MODEL_SIZE/4), activation='relu', kernel_regularizer=l2(REGULARIZER))(x)
    x = Dense(4, activation='linear')(x)
    outputs = Lambda(lambda x: x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)))(x)  # Normalisation quaternion

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the Model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        #loss='mae',  # mse Mean squared error for regression
        loss=quaternion_loss,  # Custom loss function
        metrics=['mae']  # Mean absolute error as additional metric
    )

    # Display model summary
    model.summary()
    #exit()

    
    # Save the model if validation loss improves at each epoch
    checkpoint = ModelCheckpoint("checkpoint.h5", save_best_only=True, monitor='val_loss', mode='min')
    # Stop training if validation loss does not improve after 2 epoch. Then restore the best weights
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reset_states = ResetStatesCallback([model.get_layer("lstm_1"), model.get_layer("lstm_2")])

    # Train the Model
    history = model.fit(
        train_dataset,
        epochs=EPOCH,
        validation_data=val_dataset,
        callbacks=[checkpoint, early_stopping, reset_states],
        verbose=1,
        shuffle=False  # Do not shuffle to preserve temporal order
    )

    # Evaluate on Test Set
    # Reset states before evaluation
    model.get_layer("lstm_1").reset_states()
    model.get_layer("lstm_2").reset_states()
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
    plt.savefig('loss_plot.png')
    plt.close()
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.legend()
    plt.savefig('mae_plot.png')
    plt.close()


    # Predict on Test Set
    model.get_layer("lstm_1").reset_states()
    model.get_layer("lstm_2").reset_states()
    y_pred = model.predict(test_dataset)

    print("y_pred range:", y_pred.min(), y_pred.max())

    n_samples = 2000

    plt.figure(figsize=(10, 6))
    # Tracer la première composante (par exemple, quat_w) des labels et des prédictions
    plt.plot(y_test[:, 0], label="Vérité terrain (quat_w)")
    plt.plot(y_pred[:, 0], label="Prédiction (quat_w)")
    plt.xlabel("Index de l'échantillon")
    plt.ylabel("Valeur de quat_w")
    plt.title("Comparaison des prédictions et des labels sur le jeu de test")
    plt.legend()
    plt.savefig('quat_w_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 1], label="Vérité terrain (quat_x)")
    plt.plot(y_pred[:, 1], label="Prédiction (quat_x)")
    plt.xlabel("Index de l'échantillon")
    plt.ylabel("Valeur de quat_x")
    plt.title("Comparaison des prédictions et des labels sur le jeu de test")
    plt.legend()
    plt.savefig('quat_x_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 2], label="Vérité terrain (quat_y)")
    plt.plot(y_pred[:, 2], label="Prédiction (quat_y)")
    plt.xlabel("Index de l'échantillon")
    plt.ylabel("Valeur de quat_y")
    plt.title("Comparaison des prédictions et des labels sur le jeu de test")
    plt.legend()
    plt.savefig('quat_y_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 3], label="Vérité terrain (quat_z)")
    plt.plot(y_pred[:, 3], label="Prédiction (quat_z)")
    plt.xlabel("Index de l'échantillon")
    plt.ylabel("Valeur de quat_z")
    plt.title("Comparaison des prédictions et des labels sur le jeu de test")
    plt.legend()
    plt.savefig('quat_z_plot.png')
    plt.close()


    # Generate the videos
    generate_videos(y_test, y_pred, num_videos=20, fps=20, interval=50)