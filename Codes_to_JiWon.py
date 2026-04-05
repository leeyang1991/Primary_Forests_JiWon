from lytools import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

this_root = '/shared/zhulab/YangLi/Primary_Forests_JiWon'
data_root = join(this_root,'data')

T = Tools()
class Point_Dataframe:

    def __init__(self):

        pass

    def run(self):
        self.gpkg_to_df()
        self.merge_df()
        self.add_point_tiles()
        self.reproject_point()
        self.extract_value()
        self.normalize_gfe_value()
        # self.check_extract_value()
        pass

    def gpkg_to_df(self):
        print('gpkg_to_df')
        # fpath = join(self.data_dir, f'gpkg/{self.region}_df_training_final.gpkg')
        gpkg_dir = '/gpfs/sharedfs1/zhulab/Jiwon/NSF/RF_training/df_training_gpkg_FINAL/'
        outdir = join(data_root,'pf_point/dataframe')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(gpkg_dir):
            fpath = join(gpkg_dir,f)
            region = f.split('_')[0]
            outf = join(outdir,f'{region}.df')
            if isfile(outf):
                print(f"File {outf} already exists. Skipping reprojection.")
                return
            src_crs = DIC_and_TIF().wkt_84()
            df = gpd.read_file(fpath)
            T.save_df(df,outf)
            T.df_to_excel(df,outf)

    def merge_df(self):
        print('merge_df')
        fdir = join(data_root,'pf_point/dataframe')
        outdir = join(data_root,'pf_point/dataframe_all')
        T.mkdir(outdir)
        df_list = []
        for f in tqdm(T.listdir(fdir)):
            if not(f.endswith('.df')):
                continue
            fpath = join(fdir, f)
            df = T.load_df(fpath)
            df_list.append(df)
        df_all = pd.concat(df_list, ignore_index=True)
        outf = join(outdir,'df_all.df')
        T.save_df(df_all, outf)
        T.df_to_excel(df_all,outf)


    def tiles_boundary(self):
        fdir = join(data_root,'google_alpha_earth/tiles')
        wgs_84_wkt = DIC_and_TIF().wkt_84()
        boundary_84_dict = {}
        for year in T.listdir(fdir):
            tiles_dir = join(fdir, year)
            for f in tqdm(T.listdir(tiles_dir),desc=f'{year}'):
                post_fix = f.split('.')[0].split('_')[1]
                fpath = join(tiles_dir, f)
                profile = rasterio.open(fpath).profile
                image_wkt = profile['crs'].to_wkt()
                image_boundary = RasterIO_Func().get_tif_bounds(fpath)
                image_boundary_84_list = []
                for point in image_boundary:
                    x = point[0]
                    y = point[1]
                    new_x, new_y = T.reproject_coordinates(x,y,image_wkt,wgs_84_wkt)
                    image_boundary_84_list.append((new_x, new_y))
                boundary_84_dict[post_fix] = image_boundary_84_list
        return boundary_84_dict

    def add_point_tiles(self):
        print('add_point_tiles')
        # tiles_dir = join(data_root, 'google_alpha_earth/tiles/2017')
        dff = join(data_root,'pf_point/dataframe_all/df_all.df')
        df = T.load_df(dff)
        boundary_84_dict = self.tiles_boundary()
        T.print_head_n(df)
        tile_key_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            longitude = row['longitude']
            latitude = row['latitude']
            tile_key_list_i = []
            for key in boundary_84_dict:
                boundary_84 = boundary_84_dict[key]
                is_in = self.is_point_in_boundary(longitude, latitude, boundary_84, include_edge=True)
                if is_in:
                    tile_key_list_i.append(key)
            tile_key_list_i = np.array(tile_key_list_i)
            if len(tile_key_list_i) > 0:
                tile_key_list.append(tile_key_list_i)
            else:
                tile_key_list.append(np.nan)
        df['tile_key'] = tile_key_list
        T.save_df(df, dff)
        T.df_to_excel(df,dff)

        pass

    def is_point_in_boundary(self, lon, lat, boundary, include_edge=True):
        """
        boundary format: (ll_point, lr_point, ur_point, ul_point)
        each point is (x, y) i.e., (lon, lat) in same CRS as input point
        """
        ll_point, lr_point, ur_point, ul_point = boundary
        xs = [ll_point[0], lr_point[0], ur_point[0], ul_point[0]]
        ys = [ll_point[1], lr_point[1], ur_point[1], ul_point[1]]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if include_edge:
            return (min_x <= lon <= max_x) and (min_y <= lat <= max_y)
        return (min_x < lon < max_x) and (min_y < lat < max_y)

    def get_projection_wkt(self,year,tile_key):
        fdir = join(data_root,'google_alpha_earth/tiles',year)
        fpath = join(fdir, f'{year}_{tile_key}.tif')
        with rasterio.open(fpath) as src:
            profile = src.profile
            wkt = profile['crs'].to_wkt()
            return wkt

    def reproject_point(self):
        print('reproject_point')

        outdir = join(data_root,'pf_point/reproject')
        src_crs = DIC_and_TIF().wkt_84()
        params_list = []
        for year in range(2017,2026):
            # print(year)
            params = (year,outdir,src_crs)
            params_list.append(params)
        MULTIPROCESS(self.kernel_reproject_point,params_list).run(process=9)

    def kernel_reproject_point(self,params):
        year,outdir,src_crs = params
        dff = join(data_root, 'pf_point/dataframe_all/df_all.df')
        df = T.load_df(dff)
        year = str(year)
        outdir_i = join(outdir, year)
        T.mkdir(outdir_i, True)
        outf = join(outdir_i, f'{year}.df')

        new_x_list = []
        new_y_list = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Processing {year}',mininterval=1.0):
            lon = row['longitude']
            lat = row['latitude']
            tile_key_list = row['tile_key']
            new_x_list_i = []
            new_y_list_i = []
            for tile_key in tile_key_list:
                dst_crs = self.get_projection_wkt(year, tile_key)
                new_x, new_y = T.reproject_coordinates(lon, lat, src_crs, dst_crs)
                new_x_list_i.append(new_x)
                new_y_list_i.append(new_y)
            new_x_list.append(new_x_list_i)
            new_y_list.append(new_y_list_i)
        df['new_x_gfe'] = new_x_list
        df['new_y_gfe'] = new_y_list
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def extract_value(self):
        print('extract_value')
        df_dir = join(data_root,'pf_point/reproject')
        outdir = join(data_root,'pf_point/reproject_gfe1')
        image_dir = join(data_root,'google_alpha_earth/tiles')
        T.mkdir(outdir, True)
        params_list = []
        for year in T.listdir(df_dir):
            params = (df_dir, year, image_dir, outdir)
            # self.kernel_extract_value(params)
            # exit()
            params_list.append(params)
        MULTIPROCESS(self.kernel_extract_value,params_list).run(process=9)

    def kernel_extract_value(self,params):
        df_dir,year,image_dir,outdir = params
        print(year)
        dff = join(df_dir, year, f'{year}.df')
        df = T.load_df(dff)
        # T.print_head_n(df)
        image_dir_i = join(image_dir, year)
        tile_key_list_all = []
        x_list = []
        y_list = []
        for i, row in df.iterrows():
            tile_key_list = row['tile_key']
            if len(tile_key_list) == 1:
                tile_key_list_all.append(tile_key_list[0])
                x_list.append(row['new_x_gfe'][0])
                y_list.append(row['new_y_gfe'][0])
            else:
                tile_key_list_all.append('.'.join(tile_key_list))
                x_list.append(np.nan)
                y_list.append(np.nan)
        df['tile_key_str'] = tile_key_list_all
        df['new_x_gfe_1'] = x_list
        df['new_y_gfe_1'] = y_list
        df_group_dict = T.df_groupby(df, 'tile_key_str')
        df_all_list = []
        for tile_key_str in tqdm(df_group_dict,desc=f'Processing {year}',mininterval=1.0):
            df_i = df_group_dict[tile_key_str]
            x_list_i = df_i['new_x_gfe_1'].tolist()
            y_list_i = df_i['new_y_gfe_1'].tolist()
            # print(x_list_i)
            # exit()
            image_fpath = join(image_dir_i, f'{year}_{tile_key_str}.tif')
            if not isfile(image_fpath):
                df_i['gfe_value'] = np.nan
                df_all_list.append(df_i)
                continue
            # row_list, col_list = RasterIO_Func_Extend().cal_row_col_from_coordinates(x_list_i, y_list_i,
            #                                                                          image_fpath)
            # value_list = RasterIO_Func_Extend().extract_value_from_tif_by_row_col(row_list, col_list, image_fpath)
            value_list = RasterIO_Func().extract_value_from_tif_by_x_y(x_list_i, y_list_i, image_fpath)
            df_i['gfe_value'] = list(value_list)

            df_all_list.append(df_i)
        df_all = pd.concat(df_all_list, axis=0)
        outf = join(outdir, f'{year}.df')
        T.save_df(df_all, outf)
        T.df_to_excel(df_all, outf)
        pass

    def check_extract_value(self):
        df_dir = join(data_root,'pf_point/reproject_gfe1')
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            dff = join(df_dir, f)
            df = T.load_df(dff)
            # T.print_head_n(df)
            print(f)
            print('len df', len(df))
            df = df.dropna(subset=['gfe_value'])
            print('len df', len(df))

        pass

    def normalize_gfe_value(self):
        print('normalize_gfe_value')
        df_dir = join(data_root,'pf_point/reproject_gfe1')
        outdir = join(data_root,'pf_point/reproject_gfe_normalized')
        T.mkdir(outdir, True)
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            # print(f)
            dff = join(df_dir, f)
            df = T.load_df(dff)
            df = df.dropna(subset=['gfe_value'])
            gfe_value_normalized_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{f}'):
                gfe_value = row['gfe_value']
                de_quantized_values = ((gfe_value / 127.5) ** 2) * np.sign(gfe_value)
                gfe_value_normalized_list.append(de_quantized_values)
            df['gfe_value_normalized'] = gfe_value_normalized_list
            outf = join(outdir, f)
            T.save_df(df, outf)
            T.df_to_excel(df, outf)
        pass


class cluster_analysis:

    def __init__(self):
        self.color_scheme()
        pass

    def run(self):
        df_dir = join(data_root, 'pf_point/reproject_gfe_normalized')
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            dff = join(df_dir, f)
            self.Spectrum(dff=dff, sample_n=None)
            self.plot_distance_circle(dff=dff)
            self.PCA_3d_analysis_plot(dff=dff)
            self.PCA_3d_analysis_manually_adjust_plot(dff=dff)
            self.PCA_3d_analysis_manually_adjust_shp(dff=dff)
            plt.show()

    def data_standardize(self,dff,sample_n=None):
        col_name = 'gfe_value_normalized'
        # dff = join(data_root,'landsat/extracted_point/all_region.df')
        df = T.load_df(dff)
        if sample_n is not None:
            df = df.sample(n=sample_n)
        # T.print_head_n(df)
        # exit()
        LC = T.get_df_unique_val_list(df, 'landcover_des')

        # selected LC: 'Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'
        # selected_df = df['landcover_des'].isin(
        #     ['Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'])
        selected_df = df['landcover_des'].isin(
            ['Primary dry forest', 'Primary wet forest'])
        df = df[selected_df]
        df = df.dropna(subset=[col_name])

        spectrual_values = df[col_name].tolist()
        X = np.array(spectrual_values)
        y = df['landcover_des'].tolist()
        y = np.array(y)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(dict(zip(le.classes_, range(len(le.classes_)))))
        # X_scaled = StandardScaler().fit_transform(X)
        X_scaled = X
        y_class = le.classes_
        return X_scaled, y_encoded,y_class,df

    def data_standardize_all(self,dff,sample_n=None):
        # same function as data_standardize but with all 4 classes
        col_name = 'gfe_value_normalized'
        # dff = join(data_root,'landsat/extracted_point/all_region.df')
        df = T.load_df(dff)
        if sample_n is not None:
            df = df.sample(n=sample_n)
        # T.print_head_n(df)
        # exit()
        LC = T.get_df_unique_val_list(df, 'landcover_des')

        # selected LC: 'Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'
        selected_df = df['landcover_des'].isin(
            ['Primary dry forest', 'Primary wet forest', 'Secondary dry forest', 'Secondary wet forest'])
        # selected_df = df['landcover_des'].isin(
        #     ['Primary dry forest', 'Primary wet forest'])
        df = df[selected_df]
        df = df.dropna(subset=[col_name])

        spectrual_values = df[col_name].tolist()
        X = np.array(spectrual_values)
        y = df['landcover_des'].tolist()
        y = np.array(y)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(dict(zip(le.classes_, range(len(le.classes_)))))
        # X_scaled = StandardScaler().fit_transform(X)
        X_scaled = X
        y_class = le.classes_
        return X_scaled, y_encoded,y_class,df


    def Spectrum(self,**kwargs):
        outdir = join(data_root,'pf_point/figures/spectrum')
        T.mkdir(outdir,force=True)
        X_scaled, y_encoded, y_class,df = self.data_standardize(**kwargs)
        # exit()
        dff = kwargs['dff']
        title = dff.split('/')[-1].split('.')[0]
        plt.figure(figsize=(14, 4))
        for i, label in enumerate(y_class):
            mean_spec = X_scaled[y_encoded == i].mean(axis=0)
            std_spec = X_scaled[y_encoded == i].std(axis=0)
            plt.plot(mean_spec, label=label, color=self.color_dict[label])
            # plt.fill_between(range(len(mean_spec)), mean_spec - std_spec, mean_spec + std_spec, alpha=0.2, color=self.color_dict[label])
            plt.xlabel('Bands')
            plt.ylabel('Embedding Value')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title(f'{title}')
        # plt.show()
        # outf = join(outdir, f'{title}.pdf')
        # plt.savefig(outf, dpi=900)
        # print(f"Saved spectrum plot to: {outf}")
        # plt.close()
        pass

    def plot_distance_circle(self,dff):
        out_png_dir = join(data_root,'pf_point/figures/distance_circle')
        T.mkdir(out_png_dir,force=True)
        title = dff.split('/')[-1].split('.')[0]
        from sklearn.manifold import MDS
        result_dict = self.cal_distance(dff)
        X_scaled, y_encoded, y_class,df = self.data_standardize_all(dff)
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
        # plt.imshow(D, cmap='Blues')
        # plt.colorbar()
        # plt.title("Inter-class Distance Matrix")
        # plt.xticks(ticks=range(len(y_class)), labels=y_class, rotation=45)
        # plt.yticks(ticks=range(len(y_class)), labels=y_class)
        # plt.tight_layout()
        # plt.title(f'{title}')
        # plt.show()
        # pause()

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(D)
        coords.shape = (4, 2)

        plt.figure(figsize=(5, 5))

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

        plt.title(f'{title}')

        plt.grid(alpha=0.3)
        # plt.show()
        # outf = join(out_png_dir, f'{title}.pdf')
        # plt.savefig(outf, dpi=900)
        # print(f"Saved distance circle plot to: {outf}")
        # plt.close()
        # exit()

    def cal_distance(self,dff):
        from sklearn.metrics import pairwise_distances
        X_scaled, y_encoded, y_class,df = self.data_standardize_all(dff)
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
        return result_dict

    def PCA_3d_analysis_manually_adjust_plot(self,dff):
        X_scaled, y_encoded,y_class,df = self.data_standardize(dff)
        title = dff.split('/')[-1].split('.')[0]
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        df = df.reset_index(drop=True)
        # T.print_head_n(df)
        # exit()
        # print(X_pca.shape)
        # exit()
        for i,row in tqdm(df.iterrows(), total=df.shape[0]):
            pc1 = X_pca[i,0]
            pc2 = X_pca[i,1]
            pc3 = X_pca[i,2]
            if pc1 < -0.3:
                lc_class = 'Primary dry forest'
            else:
                lc_class = 'Primary wet forest'
            df.at[i,'pc1'] = pc1
            df.at[i,'pc2'] = pc2
            df.at[i,'pc3'] = pc3
            df.at[i,'lc_class_pca'] = lc_class
        # T.print_head_n(df)
        # exit()
        idx_dry = df['lc_class_pca'] == 'Primary dry forest'
        idx_wet = df['lc_class_pca'] == 'Primary wet forest'
        pca_idx_dict = {
            'Primary dry forest': idx_dry,
            'Primary wet forest': idx_wet
        }

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(y_class):
            # idx = y_encoded == i
            idx = pca_idx_dict[label]
            ax.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                X_pca[idx, 2],
                label=label,
                s=3,
                alpha=0.3,
                color=self.color_dict[label]
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'adjusted {title}')
        # ax.set_xlim(-3,6)
        # ax.set_ylim(-3,3)
        # ax.set_zlim(-2,2)

        ax.legend()
        # plt.show()
        pass

    def PCA_3d_analysis_plot(self,dff):
        X_scaled, y_encoded,y_class,df = self.data_standardize(dff)
        # X_scaled, y_encoded,y_class,df = self.data_standardize()
        title = dff.split('/')[-1].split('.')[0]
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
                s=3,
                alpha=0.3,
                color=self.color_dict[label]
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA {title}')
        # ax.set_xlim(-3,6)
        # ax.set_ylim(-3,3)
        # ax.set_zlim(-2,2)

        ax.legend()
        # plt.show()
        pass

    def PCA_3d_analysis_manually_adjust_shp(self,dff):
        outdir = join(data_root,'pf_point/figures/pca_shp')
        T.mkdir(outdir,force=True)

        X_scaled, y_encoded,y_class,df = self.data_standardize(dff)
        title = dff.split('/')[-1].split('.')[0]
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        df = df.reset_index(drop=True)
        # T.print_head_n(df)
        # exit()
        # print(X_pca.shape)
        # exit()
        for i,row in tqdm(df.iterrows(), total=df.shape[0]):
            pc1 = X_pca[i,0]
            pc2 = X_pca[i,1]
            pc3 = X_pca[i,2]
            if pc1 < -0.3:
                lc_class = 'Primary dry forest'
            else:
                lc_class = 'Primary wet forest'
            df.at[i,'pc1'] = pc1
            df.at[i,'pc2'] = pc2
            df.at[i,'pc3'] = pc3
            df.at[i,'lc_class_pca'] = lc_class
        df = df.drop(columns=['gfe_value_normalized','gfe_value',
                              'new_x_gfe',
                              'new_y_gfe',
                              'tile_key'

                              ])
        T.print_head_n(df)
        outGPKGfn = join(outdir, f'{title}.gpkg')
        print(outGPKGfn)
        T.df_to_gpkg(df, outGPKGfn, lon_col='longitude', lat_col='latitude', layer_name='points', wkt=None)


    def color_scheme(self):

        self.color_dict = {
            'Primary dry forest': '#1f77b4',
            'Primary wet forest': '#ff7f0e',
            'Secondary dry forest': '#2ca02c',
            'Secondary wet forest': '#d62728'
        }

        self.kmean_color_dict = {
            0: '#9467bd',
            1: '#8c564b'
        }

        pass


def main():
    Point_Dataframe().run()
    cluster_analysis().run()
    pass


if __name__ == '__main__':
    main()