#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:21:23 2017

@author: Yu-Hsuan Tu

This Python Script is developed for Agisoft PhotoScan 1.3.4
Python core is 3.5.2
Update: Add compatibility of PhotoScan 1.4.0

This script runs through all chunks and will do the following:
    1. Align Photos if there's no tie point
    2. Do the standard process if there is tie point

GCP needs to be marked manually

Prerequisites for standard workflow:
    1. Set CRS
    2. Photo alignment
    3. Marking GCP
    4. Optimse Camera
    5. Set Region

The standard workflow includes:
    Build dense point cloud
    Point cloud classification
    Build model
    Build DSM
    Build DEM
    Build orthomosaic
    
All chunks will be applied.
The DEM will be generated in duplicated chunk: "chunk name"_DEM respectively
Therefore, please avoid "_DEM" in your chunk name. Otherwise, it will not be processed.
"""
import PhotoScan
import os
import numpy as np
from numpy import cos, radians
from datetime import datetime, timezone
from pysolar.solar import get_altitude, get_azimuth
from math import sqrt, atan2, degrees
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

doc = PhotoScan.app.document

###############################################################################
#
# User variables
#
# Variables for photo alignment
# Accuracy: HighestAccuracy, HighAccuracy, MediumAccuracy, LowAccuracy, LowestAccuracy
Accuracy = PhotoScan.Accuracy.HighAccuracy
Key_Limit = 40000
Tie_Limit = 10000
#
# Variables for building dense cloud
# NIR_only: True or False, whether to keep only tie points from NIR band to improve quality of trees' model
# Quality: UltraQuality, HighQuality, MediumQuality, LowQuality, LowestQuality
# Filter: AggressiveFiltering, ModerateFiltering, MildFiltering, NoFiltering
NIR_only = False
Quality = PhotoScan.Quality.HighQuality
FilterMode = PhotoScan.FilterMode.MildFiltering
#
# Variables for dense cloud ground point classification
# Maximum distance is usually twice of image resolution
# Which will be calculated later
Max_Angle = 13
Cell_Size = 10
#
# Variable for building orthomosaic
# Blending: AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending
# Color_correction: True, False
BlendingMode = PhotoScan.BlendingMode.MosaicBlending
Color_correction = True
#
# Variable for calculating date time
# UTC = True if the timestamp of image is record in UTC
# Otherwise, local time zone will be used
UTC = True
#
# Variable for sun angle calculation
# Pixelwise = True if you wish to calculate the Sun angle for every pixel
# Otherwise, The sun angle for camera centre will be used to represent whole image
# It is suggested to turn off pixelwise calculation since the differece is too small
Pixelwise = False
#
# Variable for Walthal BRDF correction
# Use for produce BRDF zenith image
BRDF = False
###############################################################################

wgs_84 = PhotoScan.CoordinateSystem('EPSG::4326')

def AlignPhoto(chunk, Accuracy, Key_Limit, Tie_Limit):
    chunk.matchPhotos(accuracy=Accuracy, 
                      generic_preselection=True, 
                      reference_preselection=True, 
                      filter_mask=False, 
                      keypoint_limit=Key_Limit, 
                      tiepoint_limit=Tie_Limit)
    chunk.alignCameras(adaptive_fitting=True)
    
def BuildDenseCloud(chunk, Quality, FilterMode):
    try:
        chunk.buildDenseCloud(quality=Quality, 
                              filter= FilterMode, 
                              keep_depth=False, 
                              reuse_depth=False)
    except:
        chunk.buildDepthMaps(quality=Quality,
                             filter=FilterMode,
                             reuse_depth=False)
        chunk.buildDenseCloud(point_colors=True)
    
def ClassifyGround(chunk, Max_Angle, Cell_Size):
    DEM_resolution, Image_resolution = GetResolution(chunk)
    chunk.dense_cloud.classifyGroundPoints(max_angle=Max_Angle, 
                                           max_distance=2*Image_resolution, 
                                           cell_size=Cell_Size)
    
def BuildModel(chunk):
    chunk.buildModel(surface=PhotoScan.SurfaceType.HeightField, 
                     interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                     face_count=PhotoScan.FaceCount.HighFaceCount, 
                     source=PhotoScan.DataSource.DenseCloudData, 
                     vertex_colors=True)
    
def BuildDSM(chunk):
    try:
        chunk.buildDem(source=PhotoScan.DataSource.DenseCloudData, 
                       interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                       projection = chunk.crs)
    except:
        chunk.buildDem(source=PhotoScan.DataSource.DenseCloudData, 
                       interpolation=PhotoScan.Interpolation.EnabledInterpolation)

def BuildDEM(chunk):
    try:
        chunk.buildDem(source=PhotoScan.DataSource.DenseCloudData, 
                       interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                       projection = chunk.crs,
                       classes=[PhotoScan.PointClass.Ground])
    except:
        chunk.buildDem(source=PhotoScan.DataSource.DenseCloudData, 
                       interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                       classes=[PhotoScan.PointClass.Ground])
    
def BuildMosaic(chunk, BlendingMode):
    try:
        chunk.buildOrthomosaic(surface=PhotoScan.DataSource.ElevationData, 
                               blending=BlendingMode, 
                               color_correction=Color_correction, 
                               fill_holes=True, 
                               projection= chunk.crs)
    except:
        if Color_correction:
            chunk.calibrateColors(source_data=PhotoScan.DataSource.ModelData, color_balance=True)
        chunk.buildOrthomosaic(surface=PhotoScan.DataSource.ElevationData, 
                               blending=BlendingMode,  
                               fill_holes=True)
    
def StandardWorkflow(doc, chunk, **kwargs):
    doc.save()
    
    # Skip the chunk if it is the DEM chunk we created
    if '_DEM' in chunk.label:
        pass
    else:
        if chunk.dense_cloud is None:
    # Disable tie points except from NIR band to improve dense clouse quality on trees
            if NIR_only:
                tie_points = chunk.point_cloud.points
                npoints = len(tie_points)
                cameras, points = GetPointMatchList(chunk, 3)
                cameras, points = CollateMatchList(cameras, points)
                enabled_points = list()
                for point_index in sorted(points.keys()):
                    enabled_points.append(int(point_index))
                disabled_points = list(set(range(npoints)) - set(enabled_points))
                for point_index in disabled_points:
                    tie_points[point_index].valid = False
                
            BuildDenseCloud(chunk, kwargs['Quality'], kwargs['FilterMode'])
    # Must save before classification. Otherwise it fails.
            doc.save()
            ClassifyGround(chunk, kwargs['Max_Angle'], kwargs['Cell_Size'])
            doc.save()
        if chunk.model is None:
            BuildModel(chunk)
        doc.save()
        
        if chunk.elevation is None:
            BuildDSM(chunk)
        
    # Because each chunk can only contain one elevation data
    # Therefore, we need to duplicate the chunk to create DEM
            new_chunk = chunk.copy(items=[PhotoScan.DataSource.DenseCloudData])
            new_chunk.label = chunk.label + '_DEM'
            doc.save()
            BuildDEM(new_chunk)
        doc.save()
        
    # Change the active chunk back
        doc.chunk = chunk
        
    # Correct BRDF effect if the option is turned on
        if chunk.orthomosaic is None:
            if BRDF is True:
                BRDFCorrection(chunk)
            BuildMosaic(chunk, kwargs['BlendingMode'])
        doc.save()

def GetResolution(chunk):
    DEM_resolution = float(chunk.dense_cloud.meta['dense_cloud/resolution']) * chunk.transform.scale
    Image_resolution = DEM_resolution / int(chunk.dense_cloud.meta['dense_cloud/depth_downscale'])
    return DEM_resolution, Image_resolution

# The following functions are for extra correction purposes
def BRDFCorrection(chunk):
        
    work_path, project_file = os.path.split(doc.path)
    project_name, ext = os.path.splitext(project_file)
    chunk_name = chunk.label
    outpath = os.path.join(work_path, '.'.join([project_name, chunk_name, 'BRDF_corrected']))
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    camera_matches = dict()
    point_matches = dict()
    Sun_zenith = dict()
    Sun_azimuth = dict()
    View_zenith = dict()
    View_azimuth = dict()
    bands = list()
    
    # Get band information
    for band in chunk.cameras[0].planes:
        bands.append(band.sensor.bands[0])
    
    # Calculate the tie points tables
    for index, band_name in enumerate(bands):
        cameras, points = GetPointMatchList(chunk, index)
        cameras, points = CollateMatchList(cameras, points)
        camera_matches[band_name] = cameras
        point_matches[band_name] = points
    
    # Calculate Sun View angles arrays for every images
    for band in [band for camera in chunk.cameras for band in camera.planes]:
        Sun_zenith[band], Sun_azimuth[band], View_zenith[band], View_azimuth[band] = \
        CreateSunViewGeometryArrays(chunk, band)
    
    # Calculate BRDF corrected image
    for band in [band for camera in chunk.cameras for band in camera.planes]:
        filename = band.photo.path
        filename_formatted = os.path.abspath(filename)
        image_path, image_nameext = os.path.split(filename)
        image_name, ext = os.path.splitext(image_nameext)
        new_path = os.path.join(outpath, ''.join([image_name, ext.lower()]))
        new_filename_formatted = os.path.abspath(os.path.join(new_path, image_nameext))
        
        if filename_formatted == new_filename_formatted:
            continue
        elif os.path.exists(new_path):
            band.photo.open(new_path, 0)
        else:
            try:
                band_name = band.sensor.bands[0]
                New_Image = BRDFImage(chunk, band, 
                                      camera_matches[band_name], point_matches[band_name], 
                                      Sun_azimuth, View_zenith, View_azimuth)
                
                New_Image.save(new_path)
                band.photo.open(new_path, 0)
                
    # If there is no tie points on either features in photo, it will raise a ValueError 
    # Disable the respective camera in this case
            except ValueError:
                band.enabled = False

def GetCameraDepth(chunk, camera):
    # Get camra depth array from camera location to elevation model
    depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)
    width = depth.width
    height = depth.height
    scale = chunk.transform.scale
    
    # Scale the depth array
    depth_scaled = PhotoScan.Image(depth.width, depth.height, ' ', 'F32')
    
    for u, v in [(u, v) for u in range(width) for v in range(height)]:
        depth_scaled[u, v] = (depth[u, v][0] * scale, )
    return depth_scaled

def GetProjectVector(u, v, camera):
    pixel = PhotoScan.Vector([u, v])
    # Calculate the vector from sensor centre to pixel (u, v)
    ray = camera.sensor.calibration.unproject(pixel)
    
    return ray

def GetPixelLocation(u, v, chunk, camera, depth_scaled, crs):
    # Get the vector from sensor centre to pixel (u, v)
    ray = camera.transform.mulv(GetProjectVector(u, v, camera))
    
    # Calculate the pixel location in chunk crs
    chunk_point = camera.center + ray * depth_scaled[u, v][0] / chunk.transform.scale
    
    # Calculate the pixel geographic location
    geo_point = crs.project(chunk.transform.matrix.mulp(chunk_point))
    
    return geo_point

def GetDateTime(camera):
    Time = camera.photo.meta['Exif/DateTimeOriginal']
    Time = datetime.strptime(Time, '%Y:%m:%d %H:%M:%S')
    if UTC is True:
    # Set time zone to UTC
        tz = timezone.utc
    else:
    # Get local time zone
        tz = datetime.now(timezone.utc).astimezone().tzinfo
        
    Time_awared = Time.replace(tzinfo=tz)
    return Time_awared

def GetSunAngle(LonLat, time):
    Sun_zenith = 90 - get_altitude(LonLat[1], LonLat[0], time)
    Sun_azimuth_South = get_azimuth(LonLat[1], LonLat[0], time)
    # Convert azimuth to zero-to-north
    Sun_azimuth = 180 - Sun_azimuth_South
    if abs(Sun_azimuth) >= 360:
        Sun_azimuth -= 360
    return Sun_zenith, Sun_azimuth

def GetViewAngle(u, v, R_t, chunk, camera):
    ray = GetProjectVector(u, v, camera)
    ray_world = R_t * ray
    View_zenith = degrees(atan2(sqrt(ray_world[0]**2 + ray_world[1]**2), ray_world[2]))
    # give y a negative sign to make 0 toward north
    # swap x and y axis since 0 degree is to north
    View_azimuth = degrees(atan2(ray_world[0], -ray_world[1]))
    # Convert negative azimuth angle to positive
    if View_azimuth < 0:
        View_azimuth += 360
    
    return View_zenith, View_azimuth

def CreateSunViewGeometryArrays(chunk, camera):
    Camera_Depth_Array = GetCameraDepth(chunk, camera)
    R_t = GetWorldRotMatrix(chunk, camera).t()
    width = camera.sensor.width
    height = camera.sensor.height
    cx = camera.sensor.calibration.cx
    cy = camera.sensor.calibration.cy
    Sun_zenith = np.zeros([height, width])
    Sun_azimuth = np.zeros([height, width])
    View_zenith = np.zeros([height, width])
    View_azimuth = np.zeros([height, width])
    DateTime = GetDateTime(camera)
    
    # Initialise the sun angle calculation for camera centre
    geo_point = GetPixelLocation(width/2+cx, width/2+cy, chunk, camera, Camera_Depth_Array, wgs_84)
    Pixel_Sun_zenith, Pixel_Sun_azimuth = GetSunAngle(geo_point, DateTime)

    for u, v in [(u, v) for u in range(width) for v in range(height)]:
    # Recalculate the Sun angle for each pixel if Pixelwise is True
        if Pixelwise is True:
            geo_point = GetPixelLocation(u, v, chunk, camera, Camera_Depth_Array, wgs_84)
            Pixel_Sun_zenith, Pixel_Sun_azimuth = GetSunAngle(geo_point, DateTime)
        
        Pixel_View_zenith, Pixel_View_azimuth = GetViewAngle(u, v, R_t, chunk, camera)
        Sun_zenith[v, u] = Pixel_Sun_zenith
        Sun_azimuth[v, u] = Pixel_Sun_azimuth
        View_zenith[v, u] = Pixel_View_zenith
        View_azimuth[v, u] = Pixel_View_azimuth
    return Sun_zenith, Sun_azimuth, View_zenith, View_azimuth

def GetWorldRotMatrix(chunk, camera):
    T = chunk.transform.matrix
    m = chunk.crs.localframe(T.mulp(camera.center))
    R = m * T * camera.transform * PhotoScan.Matrix().Diag([1, -1, -1, 1])
    return R.rotation()

def GetPointMatchList(chunk, *band):
    point_cloud = chunk.point_cloud
    points = point_cloud.points
    point_proj = point_cloud.projections
    npoints = len(points)
    camera_matches = dict()
    point_matches = dict()
    for camera in chunk.cameras:
        total = dict()
        point_index = 0
    # If no band number input, only process the master channel
        try:
            proj = point_proj[camera.planes[band[0]]]
        except IndexError:
            proj = point_proj[camera]
            
        for cur_point in proj:
            track_id = cur_point.track_id
    # Match the point track ID
            while point_index < npoints and points[point_index].track_id < track_id:
                point_index += 1
            if point_index < npoints and points[point_index].track_id == track_id:
    # Add point matches and save their pixel coordinates to list
                total[point_index] = cur_point.coord
                try:
                    point_matches[point_index][camera.planes[band[0]]] = cur_point.coord
                except KeyError:
                    point_matches[point_index] = dict()
                    try:
                        point_matches[point_index][camera.planes[band[0]]] = cur_point.coord
                    except IndexError:
                        point_matches[point_index][camera] = cur_point.coord
                except IndexError:
                    point_matches[point_index][camera] = cur_point.coord
        try:
            camera_matches[camera.planes[band[0]]] = total
        except IndexError:
            camera_matches[camera] = total
    # camera_matches describes point indice and their projected pixel coordinates for each camera
    # point_matches describes point's pixel coordinates in different cameras for each point
    return camera_matches, point_matches

def CollateMatchList(camera_matches, point_matches):
    # Keep tie points which have at least 3 observation in same band
    point_to_keep = set()
    new_camera_matches = dict()
    new_point_matches = dict()
    for point_index, value in point_matches.items():
        if len(value) >= 3:
            new_point_matches[point_index] = value
            point_to_keep.add(point_index)
    for camera, points in camera_matches.items():
        new_camera_matches[camera] = {point: coord for point, coord in iter(points.items()) if point in point_to_keep}
    return new_camera_matches, new_point_matches

def MultiLinearRegression(chunk, point_matches, Sun_azimuth, View_zenith, View_azimuth):
    cameras = point_matches.keys()
    X = list()
    Y = list()
    for camera in cameras:
        u = int(point_matches[camera][0])
        v = int(point_matches[camera][1])
        
        x1 = View_zenith[camera][v, u]
        x2 = View_azimuth[camera][v, u]
        x3 = Sun_azimuth[camera][v, u]
        y = camera.photo.image()[u, v][0]
        X.append([x1**2, x1*cos(radians(x2-x3))])
        Y.append(y)
    clf = linear_model.LinearRegression()
    model = clf.fit(X, Y)
    return model

def BRDFImage(chunk, camera, camera_matches, point_matches, Sun_azimuth, View_zenith, View_azimuth):
    width = camera.sensor.width
    height = camera.sensor.height
    
    # Because PhotoScan crashes when using numpy image IO
    # I have no choice but assign values pixelwisely
    Image = camera.photo.image()
    Image_array = np.zeros([height, width])
    for u, v in [(u, v) for u in range(width) for v in range(height)]:
        Image_array[v, u] = Image[u, v][0]
        
    datatype = {'U8':np.uint8, 'U16':np.uint16, 'U32':np.uint32, 'F32':np.float32, 'F64':np.float64}
    PhotoScan_Imtype = Image.data_type
    Imtype = datatype[PhotoScan_Imtype]
    
    # In our study, the environment usually contains only two features (ground, vegetation)
    # Therefore, we predict which feature the pixels belongs to
    # Then apply different linear model
    GMM = ImageGaussianMixture(Image_array)
    
    BRDFZenithImage = PhotoScan.Image(width, height, ' ', PhotoScan_Imtype)
    BRDFArray = [[], []]
    denoised = [[], []]
        
    # Solve BRDF for tie points
    for point_index, coords in camera_matches[camera].items():
        u = int(coords[0])
        v = int(coords[1])
        
        model = MultiLinearRegression(chunk, point_matches[point_index], Sun_azimuth, View_zenith, View_azimuth)
        
        theta = View_zenith[camera][v, u]
        phi_v = View_azimuth[camera][v, u]
        phi_s = Sun_azimuth[camera][v, u]
        y = Image[u, v][0]
        x1 = theta**2
        x2 = theta * cos(radians(phi_v-phi_s))
        feature = GMM.predict(y)[0]
        if feature == 0:
            BRDFArray[0].append([coords[0], coords[1], y, x1, x2, model.coef_[0], model.coef_[1], model.intercept_])
        else:
            BRDFArray[1].append([coords[0], coords[1], y, x1, x2, model.coef_[0], model.coef_[1], model.intercept_])
    
    # Store BRDF coefficients in different arrays for different features and denoise
    for i in range(2):
        BRDFArray[i] = np.array(BRDFArray[i])
        denoised[i] = SearchForOutliers(BRDFArray[i])
    
    # Create linear models for different features
    X0 = np.column_stack((denoised[0][:, 2], 
                          denoised[0][:, 3], 
                          denoised[0][:, 4]))
    X1 = np.column_stack((denoised[1][:, 2], 
                          denoised[1][:, 3], 
                          denoised[1][:, 4]))
    clf0 = linear_model.LinearRegression(fit_intercept=False)
    clf1 = linear_model.LinearRegression(fit_intercept=False)
    clf0.fit(X0, denoised[0][:, 7])
    clf1.fit(X1, denoised[1][:, 7])
    
    # Use known observations to predict unknown Walthal BRDF zenith observation for each pixels
    # Different linear models are applied to different features based on DN value
    features = GMM.predict(Image_array.reshape(-1, 1)).reshape(height, width)
    BRDFZenithArray = ZenithPredict(Image_array, features, 
                                    Sun_azimuth[camera], View_zenith[camera], View_azimuth[camera], 
                                    clf0, clf1)
    
    # Clip Image in the range of data type max and min
    try:
        dtype_max = np.iinfo(Imtype).max
        dtype_min = np.iinfo(Imtype).min
    except ValueError:
        dtype_max = np.finfo(Imtype).max
        dtype_min = np.finfo(Imtype).min
    np.clip(BRDFZenithArray, dtype_min, dtype_max, BRDFZenithArray)
    
    # Assign values to PhotoScan Image format pixelwisely
    for u, v in [(u, v) for u in range(width) for v in range(height)]:
        BRDFZenithImage[u, v] = (BRDFZenithArray[v, u], )
    
    return BRDFZenithImage

def SearchForOutliers(data):
    db = DBSCAN(eps=5000)
    db.fit(data[:, 5:8])
    labels = db.labels_
    # -1 means noise
    noise_mask = (labels != -1)
    denoised = data[noise_mask]
    return denoised

def ImageGaussianMixture(image):
    # image is a numpy array
    # reshape to n samples m features by reshape(n, m)
    GMM = GaussianMixture(n_components=2)
    try:
        bands = image.shape[2]
        GMM.fit(image.reshape(-1, 1, bands))
    except IndexError:
        GMM.fit(image.reshape(-1, 1))
        
    return GMM

def ZenithPredict(Image_array, features, Sun_azimuth, View_zenith, View_azimuth, clf0, clf1):
    rows, cols = Image_array.shape
    
    x1 = View_zenith ** 2
    x2 = View_zenith * cos(radians(View_azimuth - Sun_azimuth))
    samples = np.column_stack((Image_array.reshape(-1, 1),
                               x1.reshape(-1, 1),
                               x2.reshape(-1, 1)))
    
    # Create two image arrays. 
    # New_image0 is for feature 0, New_image1 is for feature1
    New_image0 = clf0.predict(samples).reshape(rows, cols)
    New_image1 = clf1.predict(samples).reshape(rows, cols)
    
    # Create a conditional vectorised function to determine which pixel values should be returned
    def func(Image0, Image1, features):
        if features == 0:
            return Image0
        else:
            return Image1
    vfunc = np.vectorize(func)
    
    New_image_array = vfunc(New_image0, New_image1, features)
    return New_image_array

# The following process will only be executed when running script    
if __name__ == '__main__':
    # Initialising listing chunks
    chunk_list = doc.chunks
        
    # Loop for all initial chunks
    for chunk in chunk_list:
        doc.chunk = chunk
        
    # Align Photo only if it is not done yet
        if chunk.point_cloud is None:
            AlignPhoto(chunk, Accuracy, Key_Limit, Tie_Limit)
            
    # Do the rest when there's tie point
        else:
            StandardWorkflow(doc, chunk, 
                             Quality=Quality, FilterMode=FilterMode, 
                             Max_Angle=Max_Angle, Cell_Size=Cell_Size, 
                             BlendingMode=BlendingMode)
    
    PhotoScan.app.update()
    
