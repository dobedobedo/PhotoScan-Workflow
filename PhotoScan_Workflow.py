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
from datetime import datetime
from pytz import utc
from pysolar import solar

doc = PhotoScan.app.document

#######################################################
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
#######################################################

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
    ray = camera.transform.mulv(camera.sensor.calibration.unproject(pixel))
    
    return ray

def GetPixelLocation(u, v, chunk, camera, depth_scaled, crs):
    # Get the vector from sensor centre to pixel (u, v)
    ray = GetProjectVector(u, v, camera)
    
    # Calculate the pixel location in chunk crs
    chunk_point = camera.center + ray * depth_scaled[u, v][0] / chunk.transform.scale
    
    # Calculate the pixel geographic location
    geo_point = crs.project(chunk.transform.matrix.mulp(chunk_point))
    
    return geo_point

def GetDateTime(camera):
    Time = camera.photo.meta['Exif/DateTimeOriginal']
    Time = datetime.strptime(Time, '%Y:%m:%d %H:%M:%S')
    Time_UTC = utc.localize(Time, is_dst=False)
    return Time_UTC

def GetSunAngle(LonLat, time):
    Sun_zenith = 90 - solar.get_altitude(LonLat[1], LonLat[0], time)
    Sun_azimuth_South = solar.get_azimuth(LonLat[1], LonLat[0], time)
    # Convert azimuth to zero-to-north
    Sun_azimuth = 180 - Sun_azimuth_South
    if abs(Sun_azimuth) >= 360:
        Sun_azimuth -= 360
    return Sun_zenith, Sun_azimuth

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
    