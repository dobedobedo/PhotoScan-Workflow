#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:21:23 2017

@author: Yu-Hsuan Tu

This Python Script is developed for Agisoft PhotoScan (current MetaShape) 1.3.4
Python core is 3.5.2

22 October 2019 Update: Add tie point error reduction following the USGS guidline
                        Add the 3D model parameters to user variables
11 January 2019 Update: Add compatibility of MetaShape 1.5.0
Update: Add compatibility of PhotoScan 1.4.0

This script runs through all chunks and will do the following:
    1. Align Photos if there's no tie point
    2. Do the standard process if there is tie point

When aligning photos, users can decide whether using image quality to disable bad photos

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
try:
    import Metashape as PhotoScan
except ImportError:
    import PhotoScan

doc = PhotoScan.app.document

#######################################################
# User variables
#
# Variables for image quality filter
# QualityFilter: True, False
# QualityCriteria: float number range from 0 to 1 (default 0.5)
QualityFilter = False
QualityCriteria = 0.5
#
# Variables for photo alignment
# Accuracy: HighestAccuracy, HighAccuracy, MediumAccuracy, LowAccuracy, LowestAccuracy
Accuracy = PhotoScan.Accuracy.HighAccuracy
Key_Limit = 60000
Tie_Limit = 0
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
# Variable for building 3D mesh
# Surface: Arbitrary, HeightField
# SurfaceSource: PointCloudData, DenseCloudData, DepthMapsData
Surface = PhotoScan.SurfaceType.Arbitrary
SurfaceSource = PhotoScan.DataSource.DepthMapsData
#
# Variable for building orthomosaic
# Since 1.4.0, users can choose performing color correction (vignetting) and balance separately.
# Blending: AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending
# Color_correction: True, False
# Color_balance: True, False
BlendingMode = PhotoScan.BlendingMode.MosaicBlending
Color_correction = True
Color_balance = False
#
#######################################################

wgs_84 = PhotoScan.CoordinateSystem("EPSG::4326")

def AlignPhoto(chunk, Accuracy, Key_Limit, Tie_Limit, QualityFilter, QualityCriteria):
    if QualityFilter:
        if chunk.cameras[0].meta['Image/Quality'] is None:
            chunk.estimateImageQuality()
        for band in [band for camera in chunk.cameras for band in camera.planes]:
            if float(band.meta['Image/Quality']) < QualityCriteria:
                band.enabled = False
    chunk.matchPhotos(accuracy=Accuracy, 
                      generic_preselection=True, 
                      reference_preselection=True, 
                      filter_mask=False, 
                      keypoint_limit=Key_Limit, 
                      tiepoint_limit=Tie_Limit)
    chunk.alignCameras()
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False, 
                          fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False, 
                          fit_p1=True, fit_p2=True, fit_p3=False, fit_p4=False, 
                          adaptive_fitting=False, tiepoint_covariance=False)
    
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
    try:
        chunk.buildModel(surface=Surface, 
                         interpolation=PhotoScan.Interpolation.EnabledInterpolation, 
                         face_count=PhotoScan.FaceCount.HighFaceCount, 
                         source=SurfaceSource, 
                         vertex_colors=True)
    except:
        chunk.buildModel(surface=Surface, 
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
            chunk.calibrateColors(source_data=PhotoScan.DataSource.ModelData, color_balance=Color_balance)
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
        
        if chunk.orthomosaic is None:
            BuildMosaic(chunk, kwargs['BlendingMode'])
        doc.save()

def GetResolution(chunk):
    DEM_resolution = float(chunk.dense_cloud.meta['dense_cloud/resolution']) * chunk.transform.scale
    Image_resolution = DEM_resolution / int(chunk.dense_cloud.meta['dense_cloud/depth_downscale'])
    return DEM_resolution, Image_resolution

def ReduceError_RU(chunk, init_threshold=10):
    # This is used to reduce error based on reconstruction uncertainty
    tie_points = chunk.point_cloud
    fltr = PhotoScan.PointCloud.Filter()
    fltr.init(chunk, PhotoScan.PointCloud.Filter.ReconstructionUncertainty)
    threshold = init_threshold
    while fltr.max_value > 10:
        fltr.selectPoints(threshold)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected >= len(tie_points.points) / 2 and threshold <= 50:
            fltr.resetSelection()
            threshold += 1
            continue
        tie_points.removeSelectedPoints()
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False, 
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False, 
                              fit_p1=True, fit_p2=True, fit_p3=False, fit_p4=False, 
                              adaptive_fitting=False, tiepoint_covariance=False)
        fltr.init(chunk, PhotoScan.PointCloud.Filter.ReconstructionUncertainty)
        threshold = init_threshold
        
def ReduceError_PA(chunk, init_threshold=2.0):
    # This is used to reduce error based on projection accuracy
    tie_points = chunk.point_cloud
    fltr = PhotoScan.PointCloud.Filter()
    fltr.init(chunk, PhotoScan.PointCloud.Filter.ProjectionAccuracy)
    threshold = init_threshold
    while fltr.max_value > 2.0:
        fltr.selectPoints(threshold)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected >= len(tie_points.points) / 2 and threshold <= 3.0:
            fltr.resetSelection()
            threshold += 0.1
            continue
        tie_points.removeSelectedPoints()
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False, 
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False, 
                              fit_p1=True, fit_p2=True, fit_p3=False, fit_p4=False, 
                              adaptive_fitting=False, tiepoint_covariance=False)
        fltr.init(chunk, PhotoScan.PointCloud.Filter.ProjectionAccuracy)
        threshold = init_threshold
    # This is to tighten tie point accuracy value
    chunk.tiepoint_accuracy = 0.1
    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True, 
                          fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True, 
                          fit_p1=True, fit_p2=True, fit_p3=True, fit_p4=True, 
                          adaptive_fitting=False, tiepoint_covariance=False)
    
def ReduceError_RE(chunk, init_threshold=0.3):
    # This is used to reduce error based on repeojection error
    tie_points = chunk.point_cloud
    fltr = PhotoScan.PointCloud.Filter()
    fltr.init(chunk, PhotoScan.PointCloud.Filter.ReprojectionError)
    threshold = init_threshold
    while fltr.max_value > 0.3:
        fltr.selectPoints(threshold)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected >= len(tie_points.points) / 10:
            fltr.resetSelection()
            threshold += 0.01
            continue
        tie_points.removeSelectedPoints()
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True, 
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True, 
                              fit_p1=True, fit_p2=True, fit_p3=True, fit_p4=True, 
                              adaptive_fitting=False, tiepoint_covariance=False)
        fltr.init(chunk, PhotoScan.PointCloud.Filter.ReprojectionError)
        threshold = init_threshold

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
            ReduceError_RU(chunk)
            ReduceError_PA(chunk)
            
    # Do the rest when there's tie point
        else:
            ReduceError_RE(chunk)
            StandardWorkflow(doc, chunk, 
                             Quality=Quality, FilterMode=FilterMode, 
                             Max_Angle=Max_Angle, Cell_Size=Cell_Size, 
                             BlendingMode=BlendingMode)
    
