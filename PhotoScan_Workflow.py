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

from Modules import sop

###############################################################################
#
# User variables
#
###############################################################################
# This section is for variables of SOP
#
# Variable to check if the script will be executed for all chunks
# all_chunk: True, False
all_chunk = False
#
# Variables for image quality filter
# QualityFilter: True, False
# QualityCriteria: float number range from 0 to 1 (default 0.5)
QualityFilter = False
QualityCriteria = 0.5
#
# Variables for photo alignment
# Accuracy: HighestAccuracy, HighAccuracy, MediumAccuracy, LowAccuracy, LowestAccuracy
Accuracy = 'HighAccuracy'
Key_Limit = 40000
Tie_Limit = 4000
#
# Variable for tie point error reduction
# error_reduction: True, False
error_reduction = False
#
# Variables for building dense cloud
# Quality: UltraQuality, HighQuality, MediumQuality, LowQuality, LowestQuality
# Filter: AggressiveFiltering, ModerateFiltering, MildFiltering, NoFiltering
Quality = 'UltraQuality'
FilterMode = ps.FilterMode.MildFiltering
#
# Variables for dense cloud ground point classification
# Maximum distance is usually twice of image resolution
# Which will be calculated later
Max_Angle = 13
Cell_Size = 10
Ero_Radius = 0.05
#
# Variable for building 3D mesh and texture
# create_model: True, False
# Surface: Arbitrary, HeightField
# SurfaceSource: PointCloudData, DenseCloudData, DepthMapsData
# uv_mapping: GenericMapping, OrthophotoMapping, AdaptiveOrthophotoMapping, SphericalMapping, CameraMapping
# texture_blending: AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending
create_model = False
Surface = ps.SurfaceType.Arbitrary
SurfaceSource = ps.DataSource.DenseCloudData
uv_mapping = ps.GenericMapping
texture_blending = ps.BlendingMode.MosaicBlending
texture_size = 4096
#
# Variable for building DSM and DEM
# ElevationSurface: PointCloudData, DenseCloudData
ElevationSource = ps.DataSource.DenseCloudData
#
# Variable for building orthomosaic
# Since 1.4.0, users can choose performing color correction (vignetting) and balance separately.
# Blending: AverageBlending, MosaicBlending, MinBlending, MaxBlending, DisabledBlending
# Color_correction: True, False
# correction_source: PointCloudData, ElevationData
# Color_balance: True, False
BlendingMode = ps.BlendingMode.MosaicBlending
Color_correction = True
correction_source = ps.DataSource.ElevationData
Color_balance = False
#
###############################################################################
# This section is for variables of extra correction
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
#

# The following process will only be executed when running script    
if __name__ == '__main__':
    doc = ps.app.document

    # Prompt the user to select CRS
    crs = ps.app.getCoordinateSystem('Select coordinate system', ps.CoordinateSystem('EPSG::4326'))

    # Run SOP workflow
    sop.run(doc, all_chunk=all_chunk, error_reduction=error_reduction, create_model=create_model,
            QualityFilter=QualityFilter, QualityCriteria=QualityCriteria, crs=crs,
            Accuracy=Accuracy, Key_Limit=Key_Limit, Tie_Limit=Tie_Limit,
            Quality=Quality, FilterMode=FilterMode,
            Max_Angle=Max_Angle, Cell_Size=Cell_Size, Ero_Radius=Ero_Radius,
            Surface=Surface, SurfaceSource=SurfaceSource,
            uv_mapping=uv_mapping, texture_blending=texture_blending, texture_size=texture_size,
            ElevationSource=ElevationSource,
            BlendingMode=BlendingMode, Color_correction=Color_correction, correction_source=correction_source,
            Color_balance=Color_balance)

    ps.app.update()
