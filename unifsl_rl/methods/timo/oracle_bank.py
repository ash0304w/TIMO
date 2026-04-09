import glob
import os


def discover_tasks(cache_root="./caches"):
    tasks = []
    for backbone in glob.glob(os.path.join(cache_root, "*")):
        for seed in glob.glob(os.path.join(backbone, "*")):
            for dataset in glob.glob(os.path.join(seed, "*")):
                tasks.append({"cache_dir": dataset})
    return tasks
