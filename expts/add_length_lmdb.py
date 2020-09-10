import os
import glob
import lmdb
import pickle
from tqdm import tqdm
import multiprocessing as mp


def connect_db(lmdb_path=None):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        meminit=False,
    )
    return env

def add_len(db_path):
    envs = connect_db(db_path)
    db_len = envs.stat()["entries"]
    txn = envs.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(db_len, protocol=-1))
    txn.commit()
    envs.sync()
    envs.close()


if __name__ == "__main__":

    lmdb_folder = (
        # f"/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/all"
        f"/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/oos_bulk/"
    )
    print(lmdb_folder)


    db_paths = glob.glob(
        os.path.join(lmdb_folder, "") + "*lmdb"
    )

    pool = mp.Pool(60)

    list(tqdm(pool.imap(add_len, db_paths), total=len(db_paths)))
