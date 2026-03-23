from lytools import *
from pyproj import Transformer


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
