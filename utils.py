from lytools import *
from pyproj import Transformer
from pprint import pprint
import pandas as pd

class RasterIO_Func_Extend(RasterIO_Func):

    def __init__(self):
        super().__init__()


    def cal_row_col_from_coordinates(self,x_list,y_list,fpath):
        row_list = []
        col_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            for i in range(len(x_list)):
                x = x_list[i]
                y = y_list[i]
                originX = profile['transform'][2]
                originY = profile['transform'][5]
                pixelWidth = profile['transform'][0]
                pixelHeight = profile['transform'][4]
                col = int((x - originX) / pixelWidth)
                row = int((y - originY) / pixelHeight)
                row_list.append(row)
                col_list.append(col)
        return row_list, col_list

    def extract_value_from_tif_by_row_col(self,row_list,col_list,fpath):
        value_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            data = src.read()
            for i in range(len(row_list)):
                col = row_list[i]
                row = col_list[i]
                value = data[:, row, col]
                value_list.append(value)
        value_list = np.array(value_list)
        return value_list

    def extract_value_from_tif_by_x_y(self,x_list,y_list,fpath):
        value_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            xy_point_list = list(zip(x_list,y_list))
            for value in src.sample(xy_point_list):
                value_list.append(value)
            value_list = np.array(value_list)
            return value_list

    def get_tif_bounds(self,fpath):
        originX_list = []
        originY_list = []
        endX_list = []
        endY_list = []
        crs_list = []
        with rasterio.open(fpath) as src:
            profile = src.profile
            crs = profile['crs']
            crs_str = crs.to_string()
            if not crs_str in crs_list:
                crs_list.append(crs_str)
            ImageHeight = profile['height']
            ImageWidth = profile['width']
            PixelWidth = profile['transform'][0]
            PixelHeight = profile['transform'][4]

            originX = profile['transform'][2]
            originY = profile['transform'][5]
            endX = originX + ImageWidth * PixelWidth
            endY = originY + ImageHeight * PixelHeight

            originX_list.append(originX)
            originY_list.append(originY)
            endX_list.append(endX)
            endY_list.append(endY)
        if len(crs_list) != 1:
            print('Different CRS found:')
            print(crs_list)
            raise ValueError('Different CRS found!')
        # originX_most = min(originX_list)
        # originY_most = max(originY_list)
        # endX_most = max(endX_list)
        # endY_most = min(endY_list)

        ll_point = (originX, originY)
        lr_point = (endX, originY)
        ur_point = (endX, endY)
        ul_point = (originX, endY)
        return ll_point,lr_point,ur_point,ul_point


class Tools_Extend(Tools):
    def __init__(self):
        super().__init__()
        pass

    def reproject_coordinates(self,x,y,src_crs,dst_crs):
        # Create a transformer object
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        # Reproject the coordinates
        new_x, new_y = transformer.transform(x, y)

        return new_x, new_y

    def shasum_string(self, input_string):
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the bytes of the input string
        sha256_hash.update(input_string.encode('utf-8'))

        # Get the hexadecimal representation of the hash
        hex_dig = sha256_hash.hexdigest()

        return hex_dig

    def point_to_shp(self, inputlist, outSHPfn,wkt=None):
        '''

        :param inputlist:

        # input list format
        # [
        # [lon,lat,{'col1':value1,'col2':value2,...}],
        #      ...,
        # [lon,lat,{'col1':value1,'col2':value2,...}]
        # ]

        :param outSHPfn:
        :return:
        '''

        fieldType_dict = {
            'float': ogr.OFTReal,
            'int': ogr.OFTInteger,
            'str': ogr.OFTString
        }

        if len(inputlist) > 0:
            if outSHPfn.endswith('.shp'):
                outSHPfn = outSHPfn
            else:
                outSHPfn = outSHPfn + '.shp'
            # Create the output shapefile
            shpDriver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(outSHPfn):
                shpDriver.DeleteDataSource(outSHPfn)
            outDataSource = shpDriver.CreateDataSource(outSHPfn)
            outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
            # Add the fields we're interested in
            col_list = inputlist[0][2].keys()
            col_list = list(col_list)
            col_list.sort()
            value_type_list = []
            for col in col_list:
                if len(col) > 10:
                    raise UserWarning(
                        f'The length of column name "{col}" is too long, length must be less than 10\nplease rename the column')
                value = inputlist[0][2][col]
                value_type = type(value)
                value_type_list.append(value_type)
            for i in range(len(value_type_list)):
                ogr_type = fieldType_dict[value_type_list[i].__name__]
                col_name = col_list[i]
                fieldDefn = ogr.FieldDefn(col_name, ogr_type)
                outLayer.CreateField(fieldDefn)
            for i in range(len(inputlist)):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(inputlist[i][0], inputlist[i][1])
                featureDefn = outLayer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(point)
                for j in range(len(col_list)):
                    col_name = col_list[j]
                    value = inputlist[i][2][col_name]
                    outFeature.SetField(col_name, value)
                # outFeature.SetField('val', inputlist[i][2])
                # 加坐标系
                spatialRef = osr.SpatialReference()
                if wkt is None:
                    wkt=DIC_and_TIF().wkt_84()
                spatialRef.ImportFromWkt(wkt)
                spatialRef.MorphToESRI()
                file = open(outSHPfn[:-4] + '.prj', 'w')
                file.write(spatialRef.ExportToWkt())
                file.close()

                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
            outFeature = None

    def df_to_shp(self, df, outSHPfn,wkt=None):
        '''

        :param inputlist:

        # input list format
        # [
        # [lon,lat,{'col1':value1,'col2':value2,...}],
        #      ...,
        # [lon,lat,{'col1':value1,'col2':value2,...}]
        # ]

        :param outSHPfn:
        :return:
        '''

        fieldType_dict = {
            'float': ogr.OFTReal,
            'int': ogr.OFTInteger,
            'str': ogr.OFTString
        }

        if len(df) > 0:
            if outSHPfn.endswith('.shp'):
                outSHPfn = outSHPfn
            else:
                outSHPfn = outSHPfn + '.shp'
            # Create the output shapefile
            shpDriver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(outSHPfn):
                shpDriver.DeleteDataSource(outSHPfn)
            outDataSource = shpDriver.CreateDataSource(outSHPfn)
            outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
            # Add the fields we're interested in
            col_list = inputlist[0][2].keys()
            col_list = list(col_list)
            col_list.sort()
            value_type_list = []
            for col in col_list:
                if len(col) > 10:
                    raise UserWarning(
                        f'The length of column name "{col}" is too long, length must be less than 10\nplease rename the column')
                value = inputlist[0][2][col]
                value_type = type(value)
                value_type_list.append(value_type)
            for i in range(len(value_type_list)):
                ogr_type = fieldType_dict[value_type_list[i].__name__]
                col_name = col_list[i]
                fieldDefn = ogr.FieldDefn(col_name, ogr_type)
                outLayer.CreateField(fieldDefn)
            for i in range(len(inputlist)):
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(inputlist[i][0], inputlist[i][1])
                featureDefn = outLayer.GetLayerDefn()
                outFeature = ogr.Feature(featureDefn)
                outFeature.SetGeometry(point)
                for j in range(len(col_list)):
                    col_name = col_list[j]
                    value = inputlist[i][2][col_name]
                    outFeature.SetField(col_name, value)
                # outFeature.SetField('val', inputlist[i][2])
                # 加坐标系
                spatialRef = osr.SpatialReference()
                if wkt is None:
                    wkt=DIC_and_TIF().wkt_84()
                spatialRef.ImportFromWkt(wkt)
                spatialRef.MorphToESRI()
                file = open(outSHPfn[:-4] + '.prj', 'w')
                file.write(spatialRef.ExportToWkt())
                file.close()

                outLayer.CreateFeature(outFeature)
                outFeature.Destroy()
            outFeature = None

    def df_to_gpkg(self, df, outGPKGfn, lon_col='lon', lat_col='lat', layer_name='points', wkt=None):
        '''
        Save a pandas DataFrame to a GeoPackage point layer.

        DataFrame must include longitude/latitude columns and any number of attribute columns.
        '''

        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas.DataFrame')
        if len(df) == 0:
            raise ValueError('df is empty')
        if lon_col not in df.columns or lat_col not in df.columns:
            raise KeyError(f'Missing required coordinate columns: {lon_col}, {lat_col}')
        if layer_name is None or str(layer_name).strip() == '':
            raise ValueError('layer_name cannot be empty')

        if outGPKGfn.endswith('.gpkg'):
            out_path = outGPKGfn
        else:
            out_path = outGPKGfn + '.gpkg'

        gpkg_driver = ogr.GetDriverByName('GPKG')
        if gpkg_driver is None:
            raise RuntimeError('GPKG driver is not available in your GDAL/OGR build')
        if os.path.exists(out_path):
            gpkg_driver.DeleteDataSource(out_path)

        out_data_source = gpkg_driver.CreateDataSource(out_path)
        if out_data_source is None:
            raise RuntimeError(f'Failed to create GeoPackage: {out_path}')

        spatial_ref = osr.SpatialReference()
        if wkt is None:
            wkt = DIC_and_TIF().wkt_84()
        spatial_ref.ImportFromWkt(wkt)

        out_layer = out_data_source.CreateLayer(str(layer_name), srs=spatial_ref, geom_type=ogr.wkbPoint)
        if out_layer is None:
            out_data_source = None
            raise RuntimeError(f'Failed to create layer: {layer_name}')

        attr_cols = [col for col in df.columns if col not in [lon_col, lat_col]]
        field_type_dict = {
            'float': ogr.OFTReal,
            'int': ogr.OFTInteger64,
            'str': ogr.OFTString,
            'bool': ogr.OFTInteger
        }

        col_type_dict = {}
        for col in attr_cols:
            series = df[col].dropna()
            if len(series) == 0:
                value_type_name = 'str'
            else:
                value = series.iloc[0]
                if isinstance(value, (bool, np.bool_)):
                    value_type_name = 'bool'
                elif isinstance(value, (int, np.integer)):
                    value_type_name = 'int'
                elif isinstance(value, (float, np.floating)):
                    value_type_name = 'float'
                else:
                    value_type_name = 'str'

            col_type_dict[col] = value_type_name
            out_layer.CreateField(ogr.FieldDefn(str(col), field_type_dict[value_type_name]))

        feature_defn = out_layer.GetLayerDefn()
        for _, row in df.iterrows():
            lon = row[lon_col]
            lat = row[lat_col]
            if pd.isna(lon) or pd.isna(lat):
                continue

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(lon), float(lat))

            out_feature = ogr.Feature(feature_defn)
            out_feature.SetGeometry(point)

            for col in attr_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                value_type_name = col_type_dict[col]
                if value_type_name == 'int':
                    out_feature.SetField(str(col), int(value))
                elif value_type_name == 'float':
                    out_feature.SetField(str(col), float(value))
                elif value_type_name == 'bool':
                    out_feature.SetField(str(col), int(bool(value)))
                else:
                    out_feature.SetField(str(col), str(value))

            out_layer.CreateFeature(out_feature)
            out_feature = None

        out_data_source = None
        return out_path

