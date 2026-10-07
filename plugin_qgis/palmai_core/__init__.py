# -*- coding: utf-8 -*-
"""
PalmAI core: QGIS-independent inference pipelines.

This package MUST NOT import qgis, and MUST NOT import heavy libraries
(torch, rasterio, ...) at import time. It is executed in three places:
  1. in-process inside QGIS (legacy "local" backend),
  2. in a dedicated virtualenv (subprocess worker),
  3. inside a Docker container (worker).
"""

JOB_TYPES = ("counting", "centertree", "classification", "road")
