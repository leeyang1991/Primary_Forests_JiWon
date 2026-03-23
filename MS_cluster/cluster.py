from lytools import *
this_root = '/home/yangli/mount/ssd4t/Primary_Forests_JiWon'
data_root = join(this_root, 'data')
T = Tools()

class Read_Point_data:
    # location at HPC: /gpfs/sharedfs1/zhulab/Jiwon/NSF/RF_training/df_training_gpkg_FINAL/

    def __init__(self):
        self.data_dir = join(data_root,'point')
        pass

    def run(self):
        self.gpkg_to_dataframe()

        pass

    def gpkg_to_dataframe(self):
        fdir = join(self.data_dir, 'gpkg')
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            df = gpd.read_file(fpath)
            T.print_head_n(df)
            df_group_dict = T.df_groupby(df,'year')
            for year in df_group_dict.keys():
                print(f'year: {year}')
            exit(0)
        # return df


class Read_Landsat:
    # location at HPC: '/gpfs/sharedfs1/zhulab/Falu/LCM_diversity/data/Landsat_composite/'
    def __init__(self):
        self.data_dir = join(data_root,'Landsat')
        pass

    def run(self):
        self.foo()
        pass

    def foo(self):
        fdir = join(self.data_dir,'HIS/HIS_median_composite_1980-1989')
        for f in T.listdir(fdir):
            print(f)


class Cluster_Analysis:
    def __init__(self):
        self.data_dir = join(data_root,'cluster')

        pass

    def run(self):

        pass

def main():
    # Read_Point_data().run()
    Read_Landsat().run()
    pass

if __name__ == '__main__':
    main()