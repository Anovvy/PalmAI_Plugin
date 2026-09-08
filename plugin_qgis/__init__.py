# -*- coding: utf-8 -*-
"""
/***************************************************************************
 PalmAI
                                 A QGIS plugin
 Mapping automation plugin for Palm Oil Plantations
 ***************************************************************************/
"""
import sys
import io

if sys.stderr is None:
    sys.stderr = io.StringIO()

if sys.stdout is None:
    sys.stdout = io.StringIO()
# --- END OF PATCH ---

def classFactory(iface):
    """"Loading the PalmAIPlugin class from the file palmai_plugin.py."""
    from .palmai_plugin import PalmAIPlugin
    return PalmAIPlugin(iface)