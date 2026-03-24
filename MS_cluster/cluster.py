import matplotlib.pyplot as plt
from utils import *
this_root = '/home/yangli/mount/ssd4t/Primary_Forests_JiWon'
data_root = join(this_root, 'data')
T = Tools_Extend()

class Read_Point_data:
    # location at HPC: /gpfs/sharedfs1/zhulab/Jiwon/NSF/RF_training/df_training_gpkg_FINAL/

    def __init__(self,region,time_range):
        self.data_dir = join(data_root,'point')
        self.region = region
        self.time_range = time_range
        pass

    def run(self):
        self.reproject_gpkg()
        pass


    def reproject_gpkg(self):
        fpath = join(self.data_dir, f'gpkg/{self.region}_df_training_final.gpkg')
        outdir = join(self.data_dir,'dataframe')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,f'{self.region}.df')
        if isfile(outf):
            print(f"File {outf} already exists. Skipping reprojection.")
            return
        src_crs = DIC_and_TIF().wkt_84()
        dst_crs = Read_Landsat(self.region,self.time_range).get_projection_wkt()

        df = gpd.read_file(fpath)
        # T.print_head_n(df)
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
    def __init__(self,region,time_range):
        self.data_dir = join(data_root,'landsat')
        self.region_lower = region.lower()
        self.region = region
        # self.time_range = '1980-1989'
        self.time_range = time_range
        pass

    def run(self):
        # self.get_projection_wkt()
        self.extract_point()
        pass

    def get_projection_wkt(self):
        fdir = join(self.data_dir, f'{self.region_lower}/{self.region}_median_composite_{self.time_range}')
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
        dff = join(Read_Point_data(self.region,self.time_range).data_dir,f'dataframe/{self.region}.df')
        df = T.load_df(dff)
        fdir = join(self.data_dir,f'{self.region_lower}/{self.region}_median_composite_{self.time_range}')
        outdir = join(self.data_dir,f'extracted_point/{self.region_lower}')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,f'{self.region}_median_composite_{self.time_range}.df')
        tile_group_df = T.df_groupby(df,'tilename')
        df_list = []
        for tile in tqdm(tile_group_df):
            tif_path = join(fdir,f'{self.region}_median_composite_{self.time_range}_{tile}.tif')
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
    def __init__(self,region,time_range):
        self.data_dir = join(data_root,'landsat/extracted_point')
        self.region = region
        self.region_lower = region.lower()
        self.time_range = time_range

        pass

    def run(self):

        self.PCA_analysis()

    def PCA_analysis(self):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        fdir = join(self.data_dir,f'{self.region_lower}')
        dff = join(fdir,f'{self.region}_median_composite_{self.time_range}.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        LC = T.get_df_unique_val_list(df,'landcover_des')
        value_list = df['value'].tolist()
        X = np.array(value_list)
        print(value_list.shape)

        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=5, alpha=0.5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA')
        plt.show()

        pass

def main():
    region_list = [
        'HIS',
        'CUBA',
        'JAM',
        'PRI',
                   ]
    time_range_list = [
        '1980-1989',
        '1990-1999',
        '2000-2009',
        '2010-2020',
    ]
    #
    for region in region_list:
        for time_range in time_range_list:
            # print(f"Processing region: {region}\n, time range: {time_range}")
            # Read_Point_data(region,time_range).run()
            # Read_Landsat(region,time_range).run()

            Cluster_Analysis(region,time_range).run()
            exit()
    pass

if __name__ == '__main__':
    main()