import matplotlib.pyplot as plt

from __global__ import *

def remove_bad_files():
    fdir = join(data_root,'google_alpha_earth/tiles')
    for year in tqdm(T.listdir(fdir)):
        year_dir = join(fdir, year)
        for f in T.listdir(year_dir):
            fpath = join(year_dir, f)
            if not f.endswith('.tif'):
                os.remove(fpath)

def count_f_num():
    fdir = join(data_root,'google_alpha_earth/tiles')
    total = 0
    for year in tqdm(T.listdir(fdir)):
        year_dir = join(fdir, year)
        flag = 0
        for f in T.listdir(year_dir):
            if not(f.endswith('.tif')):
                continue
            flag += 1
            total += 1
        print(year, flag)
    print(total)

@Decorator.shutup_gdal
def resample_embeddings():
    fdir = join(data_root,'google_alpha_earth/tiles/2018')
    outdir = join(data_root,'google_alpha_earth/tiles_resample/2018')
    T.mkdir(outdir,force=True)
    for f in tqdm(T.listdir(fdir)):
        fpath = join(fdir, f)
        outfile = join(outdir, f)
        ToRaster().resample_reproj(fpath,outfile,1000)


class Point_Dataframe:

    def __init__(self):

        pass

    def run(self):
        self.merge_df()
        self.add_point_tiles()
        self.reproject_point()
        self.extract_value()
        self.check_extract_value()
        self.normalize_gfe_value()
        pass

    def merge_df(self):

        fdir = join(data_root,'pf_point/dataframe')
        outdir = join(data_root,'pf_point/dataframe_all')
        T.mkdir(outdir)
        df_list = []
        for f in tqdm(listdir(fdir)):
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
                image_boundary = RasterIO_Func_Extend().get_tif_bounds(fpath)
                image_boundary_84_list = []
                for point in image_boundary:
                    x = point[0]
                    y = point[1]
                    new_x, new_y = T.reproject_coordinates(x,y,image_wkt,wgs_84_wkt)
                    image_boundary_84_list.append((new_x, new_y))
                boundary_84_dict[post_fix] = image_boundary_84_list
        return boundary_84_dict

    def add_point_tiles(self):
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
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Processing {year}'):
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
        for tile_key_str in tqdm(df_group_dict,desc=f'Processing {year}'):
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
            value_list = RasterIO_Func_Extend().extract_value_from_tif_by_x_y(x_list_i, y_list_i, image_fpath)
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
        df_dir = join(data_root,'pf_point/reproject_gfe1')
        outdir = join(data_root,'pf_point/reproject_gfe_normalized')
        T.mkdir(outdir, True)
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            print(f)
            dff = join(df_dir, f)
            df = T.load_df(dff)
            df = df.dropna(subset=['gfe_value'])
            gfe_value_normalized_list = []
            for i,row in df.iterrows():
                gfe_value = row['gfe_value']
                de_quantized_values = ((gfe_value / 127.5) ** 2) * np.sign(gfe_value)
                gfe_value_normalized_list.append(de_quantized_values)
            df['gfe_value_normalized'] = gfe_value_normalized_list
            outf = join(outdir, f)
            T.save_df(df, outf)
            T.df_to_excel(df, outf)
        pass


def main():
    remove_bad_files()
    # count_f_num()
    # resample_embeddings()
    Point_Dataframe().run()
    pass


if __name__ == '__main__':
    main()