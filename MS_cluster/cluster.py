import random

import matplotlib.pyplot as plt

from utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pprint import pprint

# this_root = '/home/yangli/mount/ssd4t/Primary_Forests_JiWon'
this_root = '/Users/liyang/Projects_data/Primary_Forests_JiWon'
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


class Cluster_Analysis_individual_region:
    def __init__(self,region,time_range):
        self.data_dir = join(data_root,'landsat/extracted_point')
        self.region = region
        self.region_lower = region.lower()
        self.time_range = time_range
        self.color_scheme()
        pass

    def run(self):

        # self.PCA_analysis()
        self.PCA_3d_analysis()
        # self.umap_analysis()
        # self.LDA_analysis()
        # self.distance_analysis()
        # self.parallel_coordinates_analysis()
        # self.Spectrum()

    def color_scheme(self):

        self.color_dict = {
            'Primary dry forest': '#1f77b4',
            'Primary wet forest': '#ff7f0e',
            'Secondary dry forest': '#2ca02c',
            'Secondary wet forest': '#d62728'
        }

        pass

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
        value_list_new = []
        for i,row in df.iterrows():
            value = row['value']
            if True in np.isnan(value):
                value = np.nan
            value_list_new.append(value)
        df['value'] = value_list_new
        df = df.dropna(subset=['value'])

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
                alpha=0.5,
                color=self.color_dict[label]
            )
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.title(f'{label} - PCA')
            plt.show()
        pass

    def PCA_3d_analysis(self):
        # from mpl_toolkits.mplot3d import Axes3D
        X_scaled, y_encoded,y_class = self.data_standardize()


        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(y_class):
            idx = y_encoded == i
            ax.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                X_pca[idx, 2],
                label=label,
                s=5,
                alpha=0.6,
                color=self.color_dict[label]
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA {self.region} {self.time_range}')

        ax.legend()
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
                zorder=-i,
                color=self.color_dict[label]
            )

            # plt.legend()
            plt.title(f'{label} - UMAP')
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
                # alpha=0.5,
                color=self.color_dict[label]
            )

            # plt.legend()
            plt.xlim(-3,3)
            plt.ylim(-3,3)
            plt.title(f'{label} - LDA')

            plt.show()

    def distance_analysis(self):
        from sklearn.metrics import pairwise_distances
        X_scaled, y_encoded, y_class = self.data_standardize()
        intra = []
        for i in range(4):
            intra_i = pairwise_distances(X_scaled[y_encoded == i])
            print(intra_i.shape)
        #     plt.imshow(intra_i, cmap='jet')
        #     plt.title(y_class[i])
        #     plt.colorbar()
        #     plt.show()
        #
        #     exit()
            # intra.append(intra_i.mean())

        # inter_0_1 = pairwise_distances(
        #     X_scaled[y_encoded == 0],
        #     X_scaled[y_encoded == 1]
        # ).mean()
        inter_0_2 = pairwise_distances(
            X_scaled[y_encoded == 0],
            # X_scaled[y_encoded == 1],
            # X_scaled[y_encoded == 2]
        )
        # print(inter_0_2.shape)
        # plt.imshow(inter_0_2,aspect='auto',cmap='jet')
        # plt.colorbar()
        # plt.show()
        # exit()

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
            std_spec = X_scaled[y_encoded == i].std(axis=0)
            plt.plot(mean_spec, label=label, color=self.color_dict[label])
            plt.fill_between(range(len(mean_spec)), mean_spec - std_spec, mean_spec + std_spec, alpha=0.2, color=self.color_dict[label])
            plt.xlabel('Band')
            plt.ylabel('Mean Spectral Value (Standardized)')

        plt.legend()
        plt.title("Spectrum")
        plt.show()
        pass


class Cluster_Analysis_all_region:
    def __init__(self):
        self.color_scheme()
        pass

    def run(self):
        # self.merge_df()


        # self.PCA_3d_analysis()
        # self.Spectrum()
        # self.cal_distance()
        self.plot_distance_circle()
        pass

    def data_standardize(self,sample_n=None):
        dff = join(data_root,'landsat/extracted_point/all_region.df')
        df = T.load_df(dff)
        if sample_n is not None:
            df = df.sample(n=sample_n)
        # T.print_head_n(df)
        # exit()
        LC = T.get_df_unique_val_list(df, 'landcover_des')

        # selected LC: 'Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'
        selected_df = df['landcover_des'].isin(
            ['Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'])
        df = df[selected_df]
        value_list_new = []
        for i,row in df.iterrows():
            value = row['value']
            if True in np.isnan(value):
                value = np.nan
            value_list_new.append(value)
        df['value'] = value_list_new
        df = df.dropna(subset=['value'])

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

    def PCA_3d_analysis(self):
        X_scaled, y_encoded,y_class = self.data_standardize(20000)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(y_class):
            idx = y_encoded == i
            ax.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                X_pca[idx, 2],
                label=label,
                s=5,
                alpha=0.6,
                color=self.color_dict[label]
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA All Region')
        ax.set_xlim(-3,6)
        ax.set_ylim(-3,3)
        ax.set_zlim(-2,2)

        ax.legend()
        plt.show()
        pass

    def cal_distance(self):
        from sklearn.metrics import pairwise_distances
        from sklearn.manifold import MDS
        X_scaled, y_encoded, y_class = self.data_standardize()
        intra = []
        for i in range(4):
            intra_i = pairwise_distances(X_scaled[y_encoded == i])
            intra.append(intra_i.mean())

        inter_0_1 = pairwise_distances(
            X_scaled[y_encoded == 0],
            X_scaled[y_encoded == 1]
        ).mean()
        print("inter_0_1:", inter_0_1)

        inter_0_2 = pairwise_distances(
            X_scaled[y_encoded == 0],
            X_scaled[y_encoded == 2]
        ).mean()
        print("inter_0_2:", inter_0_2)

        inter_0_3 = pairwise_distances(
            X_scaled[y_encoded == 0],
            X_scaled[y_encoded == 3]
        ).mean()
        print("inter_0_3:", inter_0_3)

        inter_1_2 = pairwise_distances(
            X_scaled[y_encoded == 1],
            X_scaled[y_encoded == 2]
        ).mean()
        print("inter_1_2:", inter_1_2)

        inter_1_3 = pairwise_distances(
            X_scaled[y_encoded == 1],
            X_scaled[y_encoded == 3]
        ).mean()
        print("inter_1_3:", inter_1_3)

        inter_2_3 = pairwise_distances(
            X_scaled[y_encoded == 2],
            X_scaled[y_encoded == 3]
        ).mean()
        print("inter_2_3:", inter_2_3)

        result_dict = {
            'intra': intra,
            'inter_0_1': inter_0_1,
            'inter_0_2': inter_0_2,
            'inter_0_3': inter_0_3,
            'inter_1_2': inter_1_2,
            'inter_1_3': inter_1_3,
            'inter_2_3': inter_2_3
        }
        pprint(result_dict)

    def distance_result(self):
        result_dict = {'inter_0_1': np.float32(2.0421948),
         'inter_0_2': np.float32(1.3460888),
         'inter_0_3': np.float32(2.707372),
         'inter_1_2': np.float32(1.8751005),
         'inter_1_3': np.float32(2.8639226),
         'inter_2_3': np.float32(2.6039674),
         'intra': [np.float32(1.471244),
                   np.float32(2.3632395),
                   np.float32(1.1304694),
                   np.float32(2.6650622)]}
        return result_dict

    def plot_distance_circle(self):
        from sklearn.manifold import MDS
        result_dict = self.distance_result()
        X_scaled, y_encoded, y_class = self.data_standardize()
        intra = result_dict['intra']
        inter_0_1 = result_dict['inter_0_1']
        inter_0_2 = result_dict['inter_0_2']
        inter_0_3 = result_dict['inter_0_3']
        inter_1_2 = result_dict['inter_1_2']
        inter_1_3 = result_dict['inter_1_3']
        inter_2_3 = result_dict['inter_2_3']
        D = np.array([
            [0, inter_0_1, inter_0_2, inter_0_3],
            [inter_0_1, 0, inter_1_2, inter_1_3],
            [inter_0_2, inter_1_2, 0, inter_2_3],
            [inter_0_3, inter_1_3, inter_2_3, 0]
        ])
        plt.imshow(D, cmap='Blues')
        plt.colorbar()
        plt.title("Inter-class Distance Matrix")
        plt.xticks(ticks=range(len(y_class)), labels=y_class, rotation=45)
        plt.yticks(ticks=range(len(y_class)), labels=y_class)
        plt.tight_layout()
        # plt.show()

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(D)
        coords.shape = (4, 2)

        plt.figure(figsize=(7, 7))

        for i in range(4):
            x, y = coords[i]
            plt.scatter(x, y, s=100)
            plt.text(x, y, y_class[i], fontsize=10)
            circle = plt.Circle(
                (x, y),
                intra[i],
                fill=False,
                linestyle='--',
                alpha=0.5
            )
            plt.gca().add_patch(circle)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.title("Class separation")

        plt.grid(alpha=0.3)
        plt.show()


    def color_scheme(self):

        self.color_dict = {
            'Primary dry forest': '#1f77b4',
            'Primary wet forest': '#ff7f0e',
            'Secondary dry forest': '#2ca02c',
            'Secondary wet forest': '#d62728'
        }

        pass

    def Spectrum(self):
        X_scaled, y_encoded, y_class = self.data_standardize()
        for i, label in enumerate(y_class):
            mean_spec = X_scaled[y_encoded == i].mean(axis=0)
            std_spec = X_scaled[y_encoded == i].std(axis=0)
            plt.plot(mean_spec, label=label, color=self.color_dict[label])
            plt.fill_between(range(len(mean_spec)), mean_spec - std_spec, mean_spec + std_spec, alpha=0.2, color=self.color_dict[label])
            plt.xlabel('Band')
            plt.ylabel('Mean Spectral Value (Standardized)')

        plt.legend()
        plt.title("Spectrum")
        plt.show()
        pass

    def merge_df(self):
        fdir = join(data_root,'landsat/extracted_point')
        df_list = []
        for region in T.listdir(fdir):
            region_dir = join(fdir, region)
            for f in T.listdir(region_dir):
                if not f.endswith('.df'):
                    continue
                fpath = join(region_dir, f)
                df = T.load_df(fpath)
                df_list.append(df)
        df_all = pd.concat(df_list,axis=0)
        outdir = join(data_root,'landsat/extracted_point')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,'all_region.df')
        T.save_df(df_all,outf)
        T.df_to_excel(df_all,outf)

        pass


def main():
    # region_list = [
    #     'HIS',
    #     'CUBA',
    #     'JAM',
    #     'PRI',
    #                ]
    # time_range_list = [
    #     '1980-1989',
    #     '1990-1999',
    #     '2000-2009',
    #     '2010-2020',
    # ]
    #
    # for region in region_list:
    #     for time_range in time_range_list:
            # if not region == 'JAM':
            # if not region == 'PRI':
            #     continue
            # print(f"region: {region}, {time_range}")

            # Read_Point_data(region,time_range).run()
            # Read_Landsat(region,time_range).run()

            # Cluster_Analysis_individual_region(region,time_range).run()
            # exit()
    Cluster_Analysis_all_region().run()
    pass

if __name__ == '__main__':
    main()