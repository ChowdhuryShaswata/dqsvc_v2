import logging

import os

from quantum_kernel_callable import dQSVC_script

from multiprocessing import Process, freeze_support
import multiprocessing as mp

if __name__ == '__main__':
    freeze_support()
    mp.set_start_method("fork")
    os.environ['"DIAG_PICKLE"'] = "1"

    #Set up logging

    logging.basicConfig(
    filename="analysis_loop_sizes.log",  # Or any desired name
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrite each run; use "a" to append instead
    )

    #Run dQSVC function
    dQSVC_script()