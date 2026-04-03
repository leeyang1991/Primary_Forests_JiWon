import sys
sys.path.append("..")
from utils import *
from pprint import pprint
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from HPC_func import *



T = Tools_Extend()
this_root = '/home/ygo26002/Project_data/Primary_Forests_JiWon'
data_root = join(this_root,'data')

class Download:

    def __init__(self):
        self.data_dir = join(data_root,'google_alpha_earth')

        pass

    def run(self):
        self.download_tiles()
        # self.check_logs()


    def check_logs(self):
        fdir = join(self.data_dir, 'log/download_tiles')
        Check_logs(fdir).read_err_files()

    def download_tiles(self):
        fpath = join(self.data_dir,'shp/alphaearth_shp.shp')
        outdir = join(self.data_dir,'tiles')
        T.mkdir(outdir)
        df = T.read_point_shp(fpath)
        T.print_head_n(df)
        # T.df_to_excel(df,fpath)
        # exit()
        # utm_zone = T.get_df_unique_val_list(df,'utm_zone')

        path_list = df['path'].tolist()
        year_list = df['year'].tolist()

        # utm_east_list = df['utm_east'].tolist()
        # utm_north_list = df['utm_north'].tolist()
        # utm_south_list = df['utm_south'].tolist()
        # utm_west_list = df['utm_west'].tolist()

        wgs_east_list = df['wgs84_east'].tolist()
        wgs_north_list = df['wgs84_nort'].tolist()
        wgs_south_list = df['wgs84_sout'].tolist()
        wgs_west_list = df['wgs84_west'].tolist()

        params_list = []
        for i in range(len(path_list)):
            path = path_list[i]
            year = year_list[i]
            east = wgs_east_list[i]
            north = wgs_north_list[i]
            south = wgs_south_list[i]
            west = wgs_west_list[i]
            params_list.append((path,year,east,north,south,west,outdir))

        job_name = 'GAE_down'
        log_folder = join(self.data_dir, 'log/download_tiles')
        init_job(job_name, params_list)
        sumbit_jobs_array(self.kernel_download_tiles, params_list, log_folder, job_name=job_name,
                          job_number_limit=10,
                          parallel_process_per_task=5,
                          slurm_array_parallelism=10,
                          parallel_process_p_or_t='t',
                          cpus_per_task=2,
                          mem_gb=4,
                          timeout_min=100,
                          slurm_partition="general",
                          # slurm_partition="debug",
                          exclude_nodes="cn[472,636]",
                          pbar_update_freq=1,
                          )
        progress_bar_monitoring(job_name)


    def kernel_download_tiles(self,params):
        import boto3
        from botocore.config import Config
        from botocore import UNSIGNED
        url,year,east,north,south,west,outdir = params
        year = str(int(year))
        pos_str = f'{east}_{north}_{south}_{west}'
        pos_str_hash = T.shasum_string(pos_str)[:8]
        outdir_i = join(outdir,year)
        outf = join(outdir_i,f'{year}_{pos_str_hash}'+'.tif')
        if isfile(outf):
            sleep(0.5)
            return
        try:
            T.mkdir(outdir_i)
        except:
            pass
        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED)
        )
        bucket = url.split('/')[2]
        key = '/'.join(url.split('/')[3:])
        s3.download_file(bucket, key, outf)

def main():
    Download().run()
    pass

if __name__ == '__main__':
    main()