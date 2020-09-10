import os
import ase.io
import shutil
import random
import subprocess
from ocdata.vasp import write_vasp_input_files

VASP_FLAGS = {
	'ibrion': 2,
	'nsw': 0,
	'isif': 0,
	'isym': 0,
	'lreal': 'Auto',
	'ediffg': -0.03,
	'symprec': 1e-10,
	'encut': 350.,
	'laechg': True,
	'lwave': False,
	'ncore': 4,
	'gga': 'RP',
	'pp': 'PBE',
	'xc': 'PBE'
}

class DFTeval:
    def __init__(self, relaxation_dir):
        self.relaxation_dir = relaxation_dir
        self.dft_path = os.path.join(relaxation_dir, "dft_files")
        os.makedirs(self.dft_path, exist_ok=True)
        self.cwd = os.getcwd()

    def write_input_files(self):
        for files in os.listdir(self.relaxation_dir):
            if files.endswith("traj"):
                new_dir = self.dft_path
                os.makedirs(new_dir, exist_ok=True)
                images = ase.io.read(os.path.join(self.relaxation_dir, files), ":")
                random.seed(234562)
                sample_images = random.sample(images, 15)
                for idx, image in enumerate(sample_images):
                    os.makedirs(new_dir, exist_ok=True)
                    subpath = os.path.join(new_dir, f"step_{idx}")
                    os.makedirs(subpath)
                    shutil.copy("/private/home/mshuaibi/baselines/ocpmodels/common/SUB_vasp.sh",
                            subpath
                    )
                    os.chdir(subpath)
                    write_vasp_input_files(image, vasp_flags=VASP_FLAGS)
                    os.chdir(self.cwd)

    def queue_jobs(self):
        script_list = []
        for subdir, dirs, files in os.walk(self.dft_path):
            for name in files:
                if name.endswith(".sh"):
                    script_list.append(os.path.join(subdir))
        for i in script_list:
            command = "sbatch"
            p = subprocess.Popen([command, "SUB_vasp.sh"], cwd=i)
            p.wait()
        return script_list
