#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:21:23 2017

@author: Yu-Hsuan Tu

This Python Script is developed for Agisoft PhotoScan 1.3.4
Python core is 3.5.2

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
from datetime import datetime, timezone
from pysolar import solar
from math import sqrt, atan2, cos, degrees, radians
from sklearn import linear_model

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
# Quality: UltraQuality, HighQuality, MediumQuality, LowQuality, LowestQuality
# Filter: AggressiveFiltering, ModerateFiltering, MildFiltering, NoFiltering
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
BlendingMode = PhotoScan.BlendingMode.MosaicBlending
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
###############################################################################

wgs_84 = PhotoScan.CoordinateSystem("EPSG::4326")

def AlignPhoto(chunk, Accuracy, Key_Limit, Tie_Limit):
    chunk.matchPhotos(accuracy=Accuracy, 
                      generic_preselection=True, 
                      reference_preselection=True, 
                      filter_mask=False, 
                      keypoint_limit=Key_Limit, 
                      tiepoint_limit=Tie_Limit)
    chunk.alignCameras(adaptive_fitting=True)
    
def BuildDenseCloud(chunk, Quality, FilterMode):
    chunk.buildDenseCloud(quality=Quality, 
                          filter= FilterMode, 
                          keep_depth=False, 
                          reuse_depth=False)
    
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
    chunk.buildDem(source=PhotoScan.DataSource.DenseCloudData, 
                   interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                   projection = chunk.crs)    

def BuildDEM(chunk):
    chunk.buildDem(source=PhotoScan.DataSource.DenseCloudData, 
                   interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                   projection = chunk.crs,
                   classes=[PhotoScan.PointClass.Ground])
    
def BuildMosaic(chunk, BlendingMode):
    chunk.buildOrthomosaic(surface=PhotoScan.DataSource.ElevationData, 
                           blending=BlendingMode, 
                           color_correction=True, 
                           fill_holes=True, 
                           projection= chunk.crs)
    
def StandardWorkflow(doc, chunk, **kwargs):
    doc.save()
    
    # Skip the chunk if it is the DEM chunk we created
    if '_DEM' in chunk.label:
        pass
    else:
        if chunk.dense_cloud is None:
            BuildDenseCloud(chunk, kwargs['Quality'], kwargs['FilterMode'])
    # Must save before classification. Otherwise it fails.
            doc.save()
            ClassifyGround(chunk, kwargs['Max_Angle'], kwargs['Cell_Size'])
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
        
        if chunk.orthomosaic is None:
            BuildMosaic(chunk, kwargs['BlendingMode'])
        doc.save()

# The following functions are for extra correction purposes

def GetResolution(chunk):
    DEM_resolution = float(chunk.dense_cloud.meta['dense_cloud/resolution']) * chunk.transform.scale
    Image_resolution = DEM_resolution / int(chunk.dense_cloud.meta['dense_cloud/depth_downscale'])
    return DEM_resolution, Image_resolution

def GetCameraDepth(chunk, camera):
    # Get camra depth array from camera location to elevation model
    depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)
    
    # Scale the depth array
    depth_scaled = PhotoScan.Image(depth.width, depth.height, " ", "F32")
    
    for y in range(depth.height):
        for x in range(depth.width):
            depth_scaled[x, y] = (depth[x, y][0] * chunk.transform.scale, )
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
    Sun_zenith = 90 - solar.get_altitude(LonLat[1], LonLat[0], time)
    Sun_azimuth_South = solar.get_azimuth(LonLat[1], LonLat[0], time)
    # Convert azimuth to zero-to-north
    Sun_azimuth = 180 - Sun_azimuth_South
    if abs(Sun_azimuth) >= 360:
        Sun_azimuth -= 360
    return Sun_zenith, Sun_azimuth

def GetViewAngle(u, v, chunk, camera):
    ray = GetProjectVector(u, v, camera)
    R = GetWorldRotMatrix(chunk, camera)
    ray_world = R.t() * ray
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
    width = camera.sensor.width
    height = camera.sensor.height
    cx = camera.sensor.calibration.cx
    cy = camera.sensor.calibration.cy
    Sun_zenith = PhotoScan.Image(width, height, " ", "F32")
    Sun_azimuth = PhotoScan.Image(width, height, " ", "F32")
    View_zenith = PhotoScan.Image(width, height, " ", "F32")
    View_azimuth = PhotoScan.Image(width, height, " ", "F32")
    
    # Initialise the sun angle calculation for camera centre
    geo_point = GetPixelLocation(width/2+cx, width/2+cy, chunk, camera, Camera_Depth_Array, wgs_84)
    Pixel_Sun_zenith, Pixel_Sun_azimuth = GetSunAngle(geo_point, GetDateTime(camera))
    
    for v in range(height):
        for u in range(width):
            
    # Recalculate the Sun angle for each pixel if Pixelwise is True
            if Pixelwise is True:
                geo_point = GetPixelLocation(u, v, chunk, camera, Camera_Depth_Array, wgs_84)
                Pixel_Sun_zenith, Pixel_Sun_azimuth = GetSunAngle(geo_point, GetDateTime(camera))
            
            Pixel_View_zenith, Pixel_View_azimuth = GetViewAngle(u, v, chunk, camera)
            Sun_zenith[u, v] = (Pixel_Sun_zenith, )
            Sun_azimuth[u, v] = (Pixel_Sun_azimuth, )
            View_zenith[u, v] = (Pixel_View_zenith, )
            View_azimuth[u, v] = (Pixel_View_azimuth, )
    return Sun_zenith, Sun_azimuth, View_zenith, View_azimuth

def GetWorldRotMatrix(chunk, camera):
    T = chunk.transform.matrix
    m = chunk.crs.localframe(T.mulp(camera.center))
    R = m * T * camera.transform * PhotoScan.Matrix().Diag([1, -1, -1, 1])
    return R.rotation()

def GetYawPitchRoll(chunk, camera):
    R = GetWorldRotMatrix(chunk, camera)
    yaw, pitch, roll = PhotoScan.utils.mat2ypr(R)
    return yaw, pitch, roll

def GetOmegaPhiKappa(chunk, camera):
    R = GetWorldRotMatrix(chunk, camera)
    omega, phi, kappa = PhotoScan.utils.mat2opk(R)
    return omega, phi, kappa

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
    # Only add valid point matches and save their pixel coordinates
                if points[point_index].valid:
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

def MultiLinearRegression(point, Sun_azimuth, View_zenith, View_azimuth):
    cameras = list(point.keys())
    X = list()
    Y = list()
    for camera in cameras:
        u = point[camera][0]
        v = point[camera][1]
        x1 = View_zenith[camera][u, v][0]
        x2 = View_azimuth[camera][u, v][0]
        x3 = Sun_azimuth[camera][u, v][0]
        y = camera.photo.image()[u, v][0]
        X.append([x1**2, x1*cos(radians(x2-x3))])
        Y.append(y)
    clf = linear_model.LinearRegression()
    model = clf.fit(X, Y)
    return model

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
    
