import submitit
from concurrent.futures import ThreadPoolExecutor
import redis
from threading import Lock
from pathos.multiprocessing import ProcessPool
import time
from rich.text import Text
import os
import pickle
from pathlib import Path
# exit()
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    ProgressColumn
)


_REDIS = None
def get_redis():
    global _REDIS
    if _REDIS is None:
        _REDIS = HPC_redis().conn_redis()
    return _REDIS


_LOCK = Lock()
_LOCAL_COUNTER = 0
def update_i(job_name, batch_size=100):
    '''
    :param job_name:
    :param Hitting Redis service every batch_size times to reduce the frequency of hitting
    '''
    global _LOCAL_COUNTER

    with _LOCK:
        _LOCAL_COUNTER += 1

        if _LOCAL_COUNTER >= batch_size:
            r = get_redis()
            r.hincrby(job_name, "step", _LOCAL_COUNTER)
            _LOCAL_COUNTER = 0


def split_into_n_jobs(lst, n_jobs):
    """
    把 lst 平均分成 n_jobs 份
    """
    total = len(lst)
    chunk_size = total // n_jobs
    remainder = total % n_jobs

    chunks = []
    start = 0

    for i in range(n_jobs):
        # 前 remainder 个 job 多分一个
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        chunks.append(lst[start:end])
        start = end
    return chunks


def mkdir(dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)


def listdir(fdir):
    '''
    Mac OS
    list the names of the files in the directory
    return sorted files list without '.DS_store'
    '''
    list_dir = []
    for f in sorted(os.listdir(fdir)):
        if f.startswith('.'):
            continue
        list_dir.append(f)
    return list_dir


class IterSpeedColumn(ProgressColumn):

    def render(self, task):
        if task.finished:
            return Text(f"{task.speed:.2f} it/s", style="green")

        speed = task.speed

        if speed is None:
            return Text("-- it/s", style="dim")

        return Text(f"{speed:.2f} it/s", style="cyan")


def sumbit_jobs_array(func, params_list, log_folder, job_name,
                      job_number_limit=500,
                      parallel_process_per_task=10,
                      slurm_array_parallelism=20,
                      parallel_process_p_or_t='p',
                      cpus_per_task=1,
                      mem_gb=1,
                      timeout_min=5,
                      slurm_partition="general",
                      exclude_nodes=None,
                      pbar_update_freq=1,
                      **kwargs
                      ):
    '''
    :param func: the kernel function to run, should take one argument, e.g. func(params)
    :param params_list: list of tuples [params1, params2, ...]
    :param log_folder: slurm log_folder
    :param job_name: slurm job_name
    :param job_number_limit: number of total jobs you want to submit to slurm
    :param parallel_process_per_task: number of parallel processes per task, Recommend equal to :param cpus_per_task
    :param slurm_array_parallelism: slurm array parallelism
    :param parallel_process_p_or_t: 'p' for multiprocessing, 't' for multi-threading
    :param cpus_per_task: number of cpus per task
    :param mem_gb: memory per task
    :param timeout_min: timeout in minutes
    :param slurm_partition: slurm partition
    :param exclude_nodes: list of nodes to exclude
    :param pbar_update_freq: frequency of updating progress bar
    '''

    if len(params_list) == 0:
        raise ValueError("params_list is empty")
    if len(params_list) > job_number_limit:
        super_params_list = split_into_n_jobs(params_list, job_number_limit)
        if parallel_process_p_or_t == 't':
            def super_func(chunk):
                def wrapper(p):
                    func(p)
                    update_i(job_name, pbar_update_freq)

                with ThreadPoolExecutor(max_workers=parallel_process_per_task) as Thread_:
                    list(Thread_.map(wrapper, chunk))

        elif parallel_process_p_or_t == 'p':
            def super_func(chunk):
                # for p in chunk:
                #     func(p)

                def wrapper(p):
                    func(p)
                    update_i(job_name, pbar_update_freq)

                pool = ProcessPool(nodes=parallel_process_per_task)
                pool.map(wrapper, chunk)

            pass
        else:
            raise ValueError("parallel_process_p_or_t must be 'p' for multiprocessing or 't' for threading")

        final_params_list = super_params_list
        final_func = super_func
    else:
        final_params_list = params_list
        final_func = func

    if os.path.exists(log_folder):
        for f in os.listdir(log_folder):
            os.remove(os.path.join(log_folder, f))
    mkdir(log_folder, force=True)

    print('submiting...')
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=job_name,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
        slurm_array_parallelism=slurm_array_parallelism,
        slurm_partition=slurm_partition,
        slurm_exclude=exclude_nodes,
        **kwargs
    )
    jobs = executor.map_array(final_func, final_params_list)

    info = {
        "Total Cores Used": cpus_per_task * slurm_array_parallelism,
        "Concurrent Processes": parallel_process_per_task * slurm_array_parallelism,
        "Total Loop Length": len(params_list),
        "Number of Jobs": len(final_params_list),
        "First Job ID": jobs[0].job_id,
        "Memory for Each Job(GB)": mem_gb,
        "Time out Minutes For Each Job": timeout_min,
        "Partition": slurm_partition,
    }
    pretty_table_print(info)


def pretty_table_print(info):
    max_key_len = max(len(k) for k in info)

    print("\n=== Job Summary ===")
    for k, v in info.items():
        print(f"{k + ':':<{max_key_len + 2}} {v}")
    print("=" * (max_key_len + 15))
    pass


class HPC_redis:

    def __init__(self):
        self.r = self.conn_redis()
        pass

    def conn_redis(self):
        redis_conf = Path.home() / '.config' / 'redis' / 'redis.conf'
        with open(redis_conf) as f:
            redis_conf = f.readlines()
            host = redis_conf[0].strip()
            port = int(redis_conf[1].strip())
            passwd = redis_conf[2].strip()

        r = redis.Redis(
            host=host,
            port=port,
            password=passwd,
        )
        # print(f"Connected to Redis at {host}:{port}")
        return r

    def hit_redis(self, job_name, amount=1):
        self.r.hincrby(name=job_name, key=job_name, amount=amount)

    def set_total_num(self, job_name, total_job: int):
        # r.delete(job_name+'_total')
        # r.hincrby(name=job_name, key=task_name+'_total', amount=total_job)
        self.r.hset(job_name, 'total', str(total_job))

    def delete_job(self, job_name):
        self.r.delete(job_name)
        pass

    def query_redis(self, job_name):
        while True:
            print(self.r.hgetall(job_name))
            time.sleep(1)


class Check_logs:
    def __init__(self, log_folder):
        self.log_folder = log_folder
        pass

    def read_err_files(self):
        log_folder = self.log_folder
        log_folder = Path(log_folder)
        err_count = 0
        for f in listdir(log_folder):
            if not f.endswith(".err"):
                continue
            fpath = log_folder / f
            # print(fpath)
            with open(fpath) as fr:
                err_content = fr.read()
                if len(err_content) != 0:
                    print('==============')
                    print(f"Error in file: {f}")
                    print(err_content)
                    err_count += 1
        print('#################')
        print(f'Total error logs: {err_count}')
        print('#################')
        pass

    def read_out_files(self):
        log_folder = self.log_folder
        log_folder = Path(log_folder)
        count = 0
        for f in listdir(log_folder):
            if not f.endswith(".out"):
                continue
            fpath = log_folder / f
            # print(fpath)
            with open(fpath) as fr:
                log_content = fr.read()
                print(log_content)
                print('------------')
                count += 1
        print(f'Total files: {count}')
        pass

    def read_result_files(self):
        log_folder = self.log_folder
        log_folder = Path(log_folder)
        count = 0
        for f in listdir(log_folder):
            if not f.endswith("_result.pkl"):
                continue
            fpath = log_folder / f
            # print(fpath)
            content = pickle.load(open(fpath, 'rb'))
            print(content)
            print('------------')
            count += 1
        print(f'Total files: {count}')
        pass

    def read_submit_files(self):
        log_folder = self.log_folder
        log_folder = Path(log_folder)
        count = 0
        for f in listdir(log_folder):
            if not f.endswith("_submitted.pkl"):
                continue
            fpath = log_folder / f
            # print(fpath)
            content = pickle.load(open(fpath, 'rb'))
            print(content)
            print('------------')
            count += 1
        print(f'Total files: {count}')
        pass


def init_job(job_name, param_list):
    total_job = len(param_list)
    r = HPC_redis().conn_redis()
    r.delete(job_name)
    r.hset(job_name, 'total', str(total_job))
    r.hset(job_name, 'step', str(0))


def progress_bar_monitoring(job_name):
    hpc_redis = HPC_redis()
    info = hpc_redis.r.hgetall(job_name)
    total = info.get(b'total')
    step = info.get(b'step')
    total_int = int(total)
    step_int = int(step)
    # exit()
    with Progress(
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} {task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            IterSpeedColumn(),
            TimeRemainingColumn(),
            # show_speed=True,
    ) as progress:
        task = progress.add_task(f"[bold green]{job_name}", total=total_int)
        while True:
            info = hpc_redis.r.hgetall(job_name)
            # print(info)
            step = info.get(b'step')
            step_int = int(step)
            progress.update(task, completed=step_int)
            if step_int >= total_int:
                break
            time.sleep(1)

