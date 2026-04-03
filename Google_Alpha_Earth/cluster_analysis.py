from __global__ import *

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
            # self.Spectrum(dff=dff, sample_n=None)
            # self.plot_distance_circle(dff=dff)
            self.PCA_3d_analysis_plot(dff=dff)

            # self.PCA_3d_analysis_manually_adjust_plot(dff=dff)
            # self.PCA_3d_analysis_manually_adjust_shp(dff=dff)
            # plt.show()

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
        outf = join(outdir, f'{title}.pdf')
        plt.savefig(outf, dpi=900)
        # print(f"Saved spectrum plot to: {outf}")
        plt.close()
        pass

    def plot_distance_circle(self,dff):
        out_png_dir = join(data_root,'pf_point/figures/distance_circle')
        T.mkdir(out_png_dir,force=True)
        title = dff.split('/')[-1].split('.')[0]
        from sklearn.manifold import MDS
        result_dict = self.cal_distance(dff)
        X_scaled, y_encoded, y_class,df = self.data_standardize(dff)
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
        outf = join(out_png_dir, f'{title}.pdf')
        plt.savefig(outf, dpi=900)
        # print(f"Saved distance circle plot to: {outf}")
        plt.close()
        # exit()

    def cal_distance(self,dff):
        from sklearn.metrics import pairwise_distances
        X_scaled, y_encoded, y_class,df = self.data_standardize(dff)
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
        plt.show()
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
        plt.show()
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
        exit()
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
        ax.set_title(title)
        # ax.set_xlim(-3,6)
        # ax.set_ylim(-3,3)
        # ax.set_zlim(-2,2)

        ax.legend()
        plt.show()
        pass


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
    cluster_analysis().run()
    pass

if __name__ == '__main__':
    main()