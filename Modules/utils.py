#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov  9 13:21:23 2017

@author: Yu-Hsuan Tu

This Python Script was originally developed for Agisoft PhotoScan (current MetaShape) 1.3.4
Python core is 3.5.2. It was later adapted to Metashape 1.8.2 with Python core 3.8.11 on March 2022.
This module contains useful utilities for Metashape workflow
"""

try:
    import Metashape as ps
except ImportError:
    import PhotoScan as ps


def GetResolution(chunk):
    try:
        DEM_resolution = float(chunk.dense_cloud.meta['dense_cloud/resolution']) * chunk.transform.scale
        Image_resolution = DEM_resolution / int(chunk.dense_cloud.meta['dense_cloud/depth_downscale'])

    except TypeError:
        DEM_resolution = float(chunk.dense_cloud.meta['BuildDenseCloud/resolution']) * chunk.transform.scale
        Image_resolution = DEM_resolution / int(chunk.dense_cloud.meta['BuildDepthMaps/downscale'])

    return DEM_resolution, Image_resolution


def ReduceError_RU(chunk, init_threshold=10):
    # This is used to reduce error based on reconstruction uncertainty
    tie_points = chunk.point_cloud
    fltr = ps.PointCloud.Filter()
    fltr.init(chunk, ps.PointCloud.Filter.ReconstructionUncertainty)
    threshold = init_threshold
    while fltr.max_value > 10:
        fltr.selectPoints(threshold)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected >= len(tie_points.points) / 2 and threshold <= 50:
            fltr.resetSelection()
            threshold += 1
            continue
        UnselectPointMatch(chunk)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected == 0:
            break
        print('Delete {} tie point(s)'.format(nselected))
        tie_points.removeSelectedPoints()
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
        fltr.init(chunk, ps.PointCloud.Filter.ReconstructionUncertainty)
        threshold = init_threshold


def ReduceError_PA(chunk, init_threshold=2.0):
    # This is used to reduce error based on projection accuracy
    tie_points = chunk.point_cloud
    fltr = ps.PointCloud.Filter()
    fltr.init(chunk, ps.PointCloud.Filter.ProjectionAccuracy)
    threshold = init_threshold
    while fltr.max_value > 2.0:
        fltr.selectPoints(threshold)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected >= len(tie_points.points) / 2 and threshold <= 3.0:
            fltr.resetSelection()
            threshold += 0.1
            continue
        UnselectPointMatch(chunk)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected == 0:
            break
        print('Delete {} tie point(s)'.format(nselected))
        tie_points.removeSelectedPoints()
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
        fltr.init(chunk, ps.PointCloud.Filter.ProjectionAccuracy)
        threshold = init_threshold
    # This is to tighten tie point accuracy value
    chunk.tiepoint_accuracy = 0.1
    try:
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True,
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True,
                              fit_p1=True, fit_p2=True, fit_p3=True, fit_p4=True,
                              adaptive_fitting=False, tiepoint_covariance=False)
    except NameError:
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True,
                              fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True,
                              fit_p1=True, fit_p2=True,
                              adaptive_fitting=False, tiepoint_covariance=False)


def ReduceError_RE(chunk, init_threshold=0.3, loop_time=2):
    # This is used to reduce error based on repeojection error
    tie_points = chunk.point_cloud
    fltr = ps.PointCloud.Filter()
    fltr.init(chunk, ps.PointCloud.Filter.ReprojectionError)
    threshold = init_threshold
    counter = 0
    while fltr.max_value > 0.3 and counter < loop_time:
        fltr.selectPoints(threshold)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected >= len(tie_points.points) / 10:
            fltr.resetSelection()
            threshold += 0.01
            continue
        UnselectPointMatch(chunk)
        nselected = len([p for p in tie_points.points if p.selected])
        if nselected == 0:
            break
        print('Delete {} tie point(s)'.format(nselected))
        tie_points.removeSelectedPoints()
        try:
            chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True,
                                  fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True,
                                  fit_p1=True, fit_p2=True, fit_p3=True, fit_p4=True,
                                  adaptive_fitting=False, tiepoint_covariance=False)
        except NameError:
            chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True,
                                  fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=True,
                                  fit_p1=True, fit_p2=True,
                                  adaptive_fitting=False, tiepoint_covariance=False)
        fltr.init(chunk, ps.PointCloud.Filter.ReprojectionError)
        threshold = init_threshold
        counter += 1


def UnselectPointMatch(chunk):
    point_cloud = chunk.point_cloud
    points = point_cloud.points
    point_proj = point_cloud.projections
    npoints = len(points)

    n_proj = dict()
    point_ids = [-1] * len(point_cloud.tracks)

    for point_id in range(0, npoints):
        point_ids[points[point_id].track_id] = point_id

    # Find the point ID using projections' track ID
    for camera in chunk.cameras:
        if camera.type != ps.Camera.Type.Regular:
            continue
        if not camera.transform:
            continue

        for proj in point_proj[camera]:
            track_id = proj.track_id
            point_id = point_ids[track_id]
            if point_id < 0:
                continue
            if not points[point_id].valid:
                continue

            if point_id in n_proj.keys():
                n_proj[point_id] += 1
            else:
                n_proj[point_id] = 1

    # Unselect points which have less than three projections
    for i in n_proj.keys():
        if n_proj[i] < 3:
            points[i].selected = False
