#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:21:23 2017

@author: Yu-Hsuan Tu

This Python Script was originally developed for Agisoft PhotoScan (current MetaShape) 1.3.4
Python core was 3.5.2.
It was later adapted to Metashape 1.8.2 with Python core 3.8.11 on March 2022.

20 March   2022 Update: Clean up the script to different modules and adapt to Metashape 1.8
21 Auguest 2020 Update: Add compatibility of Metashape 1.6
22 October 2019 Update: Add tie point error reduction following the USGS guidline
                        Add the 3D model parameters to user variables
11 January 2019 Update: Add compatibility of MetaShape 1.5.0
Update: Add compatibility of PhotoScan 1.4.0

This script runs through all chunks and will do the following:
    1. Align Photos if there's no tie point
    2. Do the rest of standard procedure if there is tie point

When aligning photos, users can decide whether using image quality to disable bad photos

Manual intervene for standard workflow:
    1. Load photos
    2. Set CRS
    3. Marking GCP
    4. Set Region

The standard workflow includes:
    Build dense point cloud
    Ground point classification
    Build model and texture (optional)
    Build DSM
    Build DEM
    Build orthomosaic

In early versions, the DEM will be generated in duplicated chunk: "chunk name"_DEM respectively
Therefore, please avoid "_DEM" in your chunk name. Otherwise, it will not be processed.
DEM will be created by duplicating DSM and build with ground point in the same chunk in supported versions.
"""

try:
    import Metashape as ps
except ImportError:
    import PhotoScan as ps

from . import utils


# Try to set the correct arguments for photo match accuracy and dense cloud quality
try:
    AccuracyDict = {'HighestAccuracy': ps.Accuracy.HighestAccuracy, 'HighAccuracy': ps.Accuracy.HighAccuracy,
                    'MediumAccuracy': ps.Accuracy.MediumAccuracy, 'LowAccuracy': ps.Accuracy.LowAccuracy,
                    'LowestAccuracy': ps.Accuracy.LowestAccuracy}
    QualityDict = {'UltraQuality': ps.Quality.UltraQuality, 'HighQuality': ps.Quality.HighQuality,
                   'MediumQuality': ps.Quality.MediumQuality, 'LowQuality': ps.Quality.LowQuality,
                   'LowestQuality': ps.Quality.LowestQuality}
except AttributeError:
    AccuracyDict = {'HighestAccuracy': 0.25, 'HighAccuracy': 1, 'MediumAccuracy': 4, 'LowAccuracy': 16, 'LowestAccuracy': 64}
    QualityDict = {'UltraQuality': 1, 'HighQuality': 4, 'MediumQuality': 16, 'LowQuality': 64, 'LowestQuality': 256}


def AlignPhoto(chunk, Accuracy, Key_Limit, Tie_Limit, QualityFilter, QualityCriteria, crs):
    Accuracy = AccuracyDict[Accuracy]
    if QualityFilter:
        if chunk.cameras[0].meta['Image/Quality'] is None:
            chunk.estimateImageQuality()
        for band in [band for camera in chunk.cameras for band in camera.planes]:
            if float(band.meta['Image/Quality']) < QualityCriteria:
                band.enabled = False

    # Perform CRS conversion
    source_crs = chunk.crs
    target_crs = crs
    if source_crs.authority != target_crs.authority:
        for camera in chunk.cameras:
            if not camera.reference.location:
                continue
            camera.reference.location = ps.CoordinateSystem.transform(camera.reference.location,
                                                                      source_crs,
                                                                      target_crs)
        chunk.crs = target_crs

    try:
        chunk.matchPhotos(accuracy=Accuracy,
                          generic_preselection=True,
                          reference_preselection=True,
                          filter_mask=False,
                          keypoint_limit=Key_Limit,
                          tiepoint_limit=Tie_Limit)
    except NameError:
        chunk.matchPhotos(downscale=Accuracy,
                          generic_preselection=True,
                          reference_preselection=True,
                          filter_mask=False,
                          keypoint_limit=Key_Limit,
                          tiepoint_limit=Tie_Limit)
    chunk.alignCameras()
    try:
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False,
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False,
                              fit_p1=True, fit_p2=True, fit_p3=False, fit_p4=False,
                              adaptive_fitting=False, tiepoint_covariance=False)
    except NameError:
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False,
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False,
                              fit_p1=True, fit_p2=True,
                              adaptive_fitting=False, tiepoint_covariance=False)


def BuildDenseCloud(chunk, Quality, FilterMode):
    Quality = QualityDict[Quality]
    try:
        chunk.buildDenseCloud(quality=Quality,
                              filter=FilterMode,
                              keep_depth=False,
                              reuse_depth=False)
    except NameError:
        try:
            chunk.buildDepthMaps(quality=Quality,
                                 filter=FilterMode,
                                 reuse_depth=False)
            chunk.buildDenseCloud(point_colors=True)
        except NameError:
            chunk.buildDepthMaps(downscale=Quality,
                                 filter_mode=FilterMode,
                                 reuse_depth=False)
            chunk.buildDenseCloud(point_colors=True)


def ClassifyGround(chunk, Max_Angle, Cell_Size, Ero_Radius):
    DEM_resolution, Image_resolution = utils.GetResolution(chunk)
    try:
        chunk.dense_cloud.classifyGroundPoints(max_angle=Max_Angle,
                                               max_distance=2 * Image_resolution,
                                               cell_size=Cell_Size,
                                               erosion_radius=Ero_Radius)
    except NameError:
        chunk.dense_cloud.classifyGroundPoints(max_angle=Max_Angle,
                                               max_distance=2 * Image_resolution,
                                               cell_size=Cell_Size)


def BuildModel(chunk, Surface, SurfaceSource, uv_mapping, texture_blending, texture_size):
    try:
        chunk.buildModel(surface=Surface,
                         interpolation=ps.Interpolation.EnabledInterpolation,
                         face_count=ps.FaceCount.HighFaceCount,
                         source=SurfaceSource,
                         vertex_colors=True)

    except NameError:
        chunk.buildModel(surface_type=Surface,
                         interpolation=ps.Interpolation.EnabledInterpolation,
                         face_count=ps.FaceCount.HighFaceCount,
                         source_data=SurfaceSource,
                         vertex_colors=True)

    chunk.buildUV(mapping_mode=uv_mapping)
    chunk.buildTexture(blending_mode=texture_blending, texture_size=texture_size)


def BuildDSM(chunk, ElevationSource, crs):
    try:
        chunk.buildDem(source=ElevationSource,
                       interpolation=ps.Interpolation.EnabledInterpolation,
                       projection=crs)

    except NameError:
        proj = ps.OrthoProjection()
        proj.crs = crs
        chunk.buildDem(source_data=ElevationSource,
                       interpolation=ps.Interpolation.EnabledInterpolation,
                       projection=proj)


def BuildDEM(chunk, ElevationSource, crs):
    try:
        chunk.buildDem(source=ElevationSource,
                       interpolation=ps.Interpolation.EnabledInterpolation,
                       projection=crs,
                       classes=[ps.PointClass.Ground])

    except NameError:
        proj = ps.OrthoProjection()
        proj.crs = crs
        chunk.buildDem(source_data=ElevationSource,
                       interpolation=ps.Interpolation.EnabledInterpolation,
                       projection=proj,
                       classes=[ps.PointClass.Ground])


def BuildMosaic(chunk, BlendingMode, Color_correction, correction_source, Color_balance, crs):
    # Reset color correction
    for sensor in chunk.sensors:
        sensor.vignetting = []
    for camera in chunk.cameras:
        camera.vignetting = []

    try:
        chunk.buildOrthomosaic(surface=ps.DataSource.ElevationData,
                               blending=BlendingMode,
                               color_correction=Color_correction,
                               fill_holes=True,
                               projection=chunk.crs)
    except NameError:
        proj = ps.OrthoProjection()
        proj.crs = crs
        if Color_correction:
            try:
                chunk.calibrateColors(source_data=correction_source, color_balance=Color_balance)
            except NameError:
                chunk.calibrateColors(source_data=correction_source, white_balance=Color_balance)
        try:
            chunk.buildOrthomosaic(surface=ps.DataSource.ElevationData,
                                   blending=BlendingMode,
                                   fill_holes=True,
                                   projection=proj)
        except NameError:
            chunk.buildOrthomosaic(surface_data=ps.DataSource.ElevationData,
                                   blending_mode=BlendingMode,
                                   fill_holes=True,
                                   projection=proj)


def run(doc, all_chunk, error_reduction, create_model, **kwargs):
    doc.save()
    if all_chunk:
        chunk_list = doc.chunks
    else:
        chunk_list = [doc.chunk]

    # Check if copy method is available for Elevation object
    # If not, create separate chunk for DEM
    if hasattr(ps.Elevation, 'copy') and callable(getattr(ps.Elevation, 'copy')):
        DEM_chunk = False
    else:
        DEM_chunk = True

    for chunk in chunk_list:
        doc.chunk = chunk

        if chunk.point_cloud is None:
            AlignPhoto(chunk, Accuracy=kwargs['Accuracy'], Key_Limit=kwargs['Key_Limit'], Tie_Limit=kwargs['Tie_Limit'],
                       QualityFilter=kwargs['QualityFilter'], QualityCriteria=kwargs['QualityCriteria'],
                       crs=kwargs['crs'])
            if error_reduction:
                utils.ReduceError_RU(chunk)
                utils.ReduceError_PA(chunk)

        else:

            # Skip the chunk if DEM chunk is needed
            if DEM_chunk and ('_DEM' in chunk.label):
                pass
            else:
                if error_reduction:
                    utils.ReduceError_RE(chunk)
                    # Optimise the camera again with model C
                    try:
                        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False,
                                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False,
                                              fit_p1=True, fit_p2=True, fit_p3=False, fit_p4=False,
                                              adaptive_fitting=False, tiepoint_covariance=False)
                    except NameError:
                        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False,
                                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False,
                                              fit_p1=True, fit_p2=True,
                                              adaptive_fitting=False, tiepoint_covariance=False)

                if chunk.dense_cloud is None:
                    BuildDenseCloud(chunk, Quality=kwargs['Quality'], FilterMode=kwargs['FilterMode'])
                    # Must save before classification. Otherwise it fails.
                    doc.save()
                    ClassifyGround(chunk, Max_Angle=kwargs['Max_Angle'], Cell_Size=kwargs['Cell_Size'],
                                   Ero_Radius=kwargs['Ero_Radius'])
                    doc.save()
                if create_model and chunk.model is None:
                    BuildModel(chunk, Surface=kwargs['Surface'], SurfaceSource=kwargs['SurfaceSource'],
                               uv_mapping=kwargs['uv_mapping'], texture_blending=kwargs['texture_blending'],
                               texture_size=kwargs['texture_size'])
                doc.save()

                if chunk.elevation is None:
                    BuildDSM(chunk, ElevationSource=kwargs['ElevationSource'], crs=kwargs['crs'])
                    chunk.elevation.label = 'DSM'
                    chunk_DSM = chunk.elevation

                    # Create DEM chunk if needed
                    # Otherwise, create a copy of DEM in the same chunk if possible
                    if DEM_chunk:
                        new_chunk = chunk.copy(items=[ps.DataSource.DenseCloudData])
                        new_chunk.label = chunk.label + '_DEM'
                        doc.save()
                        BuildDEM(new_chunk, ElevationSource=kwargs['ElevationSource'], crs=kwargs['crs'])
                    else:
                        chunk.elevation.copy()
                        chunk.elevations[-1].label = 'DEM'
                        chunk.elevation = chunk.elevations[-1]
                        doc.save()
                        BuildDEM(chunk, ElevationSource=kwargs['ElevationSource'], crs=kwargs['crs'])
                        # Set DSM to default elevation object for orthomosaic creation
                        chunk.elevation = chunk_DSM

                doc.save()

                # Change the active chunk back
                doc.chunk = chunk

                if chunk.orthomosaic is None:
                    BuildMosaic(chunk, BlendingMode=kwargs['BlendingMode'], Color_correction=kwargs['Color_correction'],
                                correction_source=kwargs['correction_source'], Color_balance=kwargs['Color_balance'],
                                crs=kwargs['crs'])
                doc.save()
