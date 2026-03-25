import random

import matplotlib.pyplot as plt

from utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

        # self.PCA_analysis()
        # self.umap_analysis()
        # self.LDA_analysis()
        self.distance_analysis()
        # self.parallel_coordinates_analysis()
        # self.Spectrum()

    def data_standardize(self):
        fdir = join(self.data_dir, f'{self.region_lower}')
        dff = join(fdir, f'{self.region}_median_composite_{self.time_range}.df')
        df = T.load_df(dff)
        # T.print_head_n(df)
        # exit()
        LC = T.get_df_unique_val_list(df, 'landcover_des')

        # selected LC: 'Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'
        selected_df = df['landcover_des'].isin(
            ['Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'])
        df = df[selected_df]

        spectrual_values = df['value'].tolist()
        X = np.array(spectrual_values)
        y = df['landcover_des'].tolist()
        y = np.array(y)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(dict(zip(le.classes_, range(len(le.classes_)))))
        X_scaled = StandardScaler().fit_transform(X)
        y_class = le.classes_
        return X_scaled, y_encoded,y_class


    def PCA_analysis(self):
        X_scaled, y_encoded,y_class = self.data_standardize()


        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure()
        for i, label in enumerate(y_class):
            plt.scatter(
                X_pca[y_encoded == i, 0],
                X_pca[y_encoded == i, 1],
                label=label,
                s=5,
                alpha=0.5
            )
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA - 4 classes')
        plt.show()
        pass

    def umap_analysis(self):
        seed = random.seed(42)
        X_scaled, y_encoded, y_class = self.data_standardize()
        X_umap = umap.UMAP(n_neighbors=30, min_dist=0.1,random_state=42).fit_transform(X_scaled)

        plt.figure()
        for i, label in enumerate(y_class):
            plt.scatter(
                X_umap[y_encoded == i, 0],
                X_umap[y_encoded == i, 1],
                label=label,
                s=15,
                alpha=0.3,
                zorder=-i
            )

            # plt.legend()
            plt.title(label)
            plt.show()

    def LDA_analysis(self):
        seed = random.seed(42)
        X_scaled, y_encoded, y_class = self.data_standardize()
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X_scaled, y_encoded)

        plt.figure()
        for i, label in enumerate(y_class):
            plt.scatter(
                X_lda[y_encoded == i, 0],
                X_lda[y_encoded == i, 1],
                label=label,
                s=5,
                alpha=0.5
            )

            # plt.legend()
            plt.xlim(-3,3)
            plt.ylim(-3,3)
            plt.title(label)
            plt.show()

    def distance_analysis(self):
        from sklearn.metrics import pairwise_distances
        X_scaled, y_encoded, y_class = self.data_standardize()
        intra = []
        for i in range(4):
            intra_i = pairwise_distances(X_scaled[y_encoded == i])
            print(intra_i.shape)
            plt.imshow(intra_i, cmap='jet')
            plt.title(y_class[i])
            plt.colorbar()
            plt.show()

            exit()
            # intra.append(intra_i.mean())

        inter_0_1 = pairwise_distances(
            X_scaled[y_encoded == 0],
            X_scaled[y_encoded == 1]
        ).mean()
        inter_0_2 = pairwise_distances(
            X_scaled[y_encoded == 0],
            X_scaled[y_encoded == 2]
        )
        print(inter_0_2.shape)
        plt.imshow(inter_0_2,aspect='auto',cmap='gray')
        plt.colorbar()
        plt.show()
        exit()

        intra = [float(i) for i in intra]
        print("intra:", intra)
        print("inter:", inter_0_2)

        pass

    def parallel_coordinates_analysis(self):
        import pandas as pd
        from pandas.plotting import parallel_coordinates
        X_scaled, y_encoded, y_class = self.data_standardize()
        df = pd.DataFrame(X_scaled, columns=[f'B{i}' for i in range(1,8)])
        df['label'] = y_encoded

        parallel_coordinates(df.sample(1000), 'label', alpha=0.2)
        plt.title("Parallel coordinates (7D space)")
        plt.show()

        pass

    def Spectrum(self):
        X_scaled, y_encoded, y_class = self.data_standardize()
        for i, label in enumerate(y_class):
            mean_spec = X_scaled[y_encoded == i].mean(axis=0)
            plt.plot(mean_spec, label=label)

        plt.legend()
        plt.title("Spectrum")
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