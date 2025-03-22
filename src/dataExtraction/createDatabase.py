# Import from third party
import h5py
import sqlite3
import numpy as np

# Import from STL
import os


"""
Get the keys of the file
@param file: str - path to the file
@return list of keys
"""
def getKeys(file: str) -> list[str]:
    _keys: list[str] = []
    with h5py.File(file, 'r') as f:
        def visit(name, obj):
            _keys.append(name)

        f.visititems(visit)

    return _keys




def processFile(file: str) -> None:
    """
    Process the file
    @param file: str - path to the file
    @return None
    """

    keys: list[str] = getKeys(file)
    print(keys)

    with h5py.File(file, 'r') as f:
        for key in keys:
            data = f[key][:]  # read the full dataset into a NumPy array
            print(data.shape)
            print(data[:5])  # print first 5 rows


def is_hdf5_file(path: str) -> bool:
    """
    Check if the file is an HDF5 file
    @param path: str - path to the file
    @return bool - True if the file is an HDF5 file, False otherwise
    """
    return h5py.is_hdf5(path)



def createDatabase(folder_path: str) -> None:
    """
    Create the SQLite database
    @param folder_path: str - path to the folder containing the HDF5 files
    @return None
    """

    # HDF5 files to process
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(files)

    # SQLite database to create
    sqlite_db: str = "imu_data.db"

    # Create SQLite connection
    conn: sqlite3.Connection = sqlite3.connect(sqlite_db)
    cursor: sqlite3.Cursor = conn.cursor()

    # Create table schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS imu_data (
            file_id TEXT,
            sample_idx INTEGER,
            acc_x REAL, acc_y REAL, acc_z REAL,
            gyr_x REAL, gyr_y REAL, gyr_z REAL,
            mag_x REAL, mag_y REAL, mag_z REAL,
            pos_x REAL, pos_y REAL, pos_z REAL,
            quat_w REAL, quat_x REAL, quat_y REAL, quat_z REAL,
            movement INTEGER
        )
    ''')

    # Process each file
    for file_path in files:

        file_id = os.path.basename(file_path)

        # Skip non-HDF5 files
        if (not is_hdf5_file(file_path)):
                print(f"Skipping {file_id}...")
                continue
        
        with h5py.File(file_path, 'r') as f:

            # Extract data from HDF5 file
            acc: np.ndarray = f['imu_acc'][:]
            gyr: np.ndarray = f['imu_gyr'][:]
            mag: np.ndarray = f['imu_mag'][:]
            pos: np.ndarray = f['opt_pos'][:]
            quat: np.ndarray = f['opt_quat'][:]
            move: np.ndarray = f['movement'][:]

            N = acc.shape[0]  # number of samples
            print(f"Processing {file_id} with {N} samples...")

            # Insert data into SQLite database
            for i in range(N):
                row: tuple = (
                    file_id,
                    i,
                    *acc[i], *gyr[i], *mag[i],
                    *pos[i], *quat[i],
                    int(move[i])
                )

                # Inserts one row of data into the imu_data table
                cursor.execute('''
                    INSERT INTO imu_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', row)

    # Commit and close
    conn.commit()
    conn.close()

    print("All data stored in SQLite database.")



if __name__ == "__main__":

    # Path to the folder containing the HDF5 files
    folder_path = "../../submodules/broad/data_hdf5"

    # Create the SQLite database
    createDatabase(folder_path)
