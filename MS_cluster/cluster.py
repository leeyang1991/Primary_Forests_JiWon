import matplotlib.pyplot as plt
from utils import *
this_root = '/home/yangli/mount/ssd4t/Primary_Forests_JiWon'
data_root = join(this_root, 'data')
T = Tools_Extend()

class Read_Point_data:
    # location at HPC: /gpfs/sharedfs1/zhulab/Jiwon/NSF/RF_training/df_training_gpkg_FINAL/

    def __init__(self):
        self.data_dir = join(data_root,'point')
        pass

    def run(self):
        # self.gpkg_to_dataframe()
        self.reproject_gpkg()

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

    def reproject_gpkg(self):
        fpath = join(self.data_dir, 'gpkg/HIS_df_training_final.gpkg')
        outdir = join(self.data_dir,'dataframe')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'HIS.df')
        src_crs = DIC_and_TIF().wkt_84()
        dst_crs = Read_Landsat().get_projection_wkt()

        df = gpd.read_file(fpath)
        T.print_head_n(df)
        new_x_list = []
        new_y_list = []
        for i,row in tqdm(df.iterrows(),total=df.shape[0]):
            lon = row['longitude']
            lat = row['latitude']
            new_x, new_y = T.reproject_coordinates(lon, lat, src_crs, dst_crs)
            new_x_list.append(new_x)
            new_y_list.append(new_y)
        df['new_x'] = new_x_list
        df['new_y'] = new_y_list
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        pass


class Read_Landsat:
    # location at HPC: '/gpfs/sharedfs1/zhulab/Falu/LCM_diversity/data/Landsat_composite/'
    def __init__(self):
        self.data_dir = join(data_root,'Landsat')
        pass

    def run(self):
        # self.get_projection_wkt()
        self.extract_point()
        pass

    def get_projection_wkt(self):
        fdir = join(self.data_dir, 'HIS/HIS_median_composite_1980-1989')
        wkt_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            with rasterio.open(fpath) as src:
                profile = src.profile
                wkt = profile['crs'].to_wkt()
                wkt_list.append(wkt)
        unique_wkt = list(set(wkt_list))
        if len(unique_wkt) == 1:
            return unique_wkt[0]
        else:
            raise ValueError("Multiple unique WKT found in the files.")

    def extract_point(self):
        dff = join(Read_Point_data().data_dir,'dataframe/HIS.df')
        df = T.load_df(dff)
        fdir = join(self.data_dir,'HIS/HIS_median_composite_1980-1989')
        outdir = join(self.data_dir,'HIS/extracted_point')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'HIS_median_composite_1980-1989.df')
        tile_group_df = T.df_groupby(df,'tilename')
        df_list = []
        for tile in tqdm(tile_group_df):
            tif_path = join(fdir,f'HIS_median_composite_1980-1989_{tile}.tif')
            df_i = tile_group_df[tile]
            new_x_list = df_i['new_x'].tolist()
            new_y_list = df_i['new_y'].tolist()
            row_list, col_list = RasterIO_Func_Extend().cal_row_col_from_coordinates(new_x_list,new_y_list,tif_path)
            value_list = RasterIO_Func_Extend().extract_value_from_tif_by_row_col(row_list,col_list,tif_path)
            df_i['value'] = list(value_list)
            df_list.append(df_i)
        df_all = pd.concat(df_list,axis=0)
        T.save_df(df_all,outf)
        T.df_to_excel(df_all,outf)


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