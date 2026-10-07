# -*- coding: utf-8 -*-
"""Helpers that load job results into QGIS (main thread only)."""

from qgis.core import Qgis, QgsMessageLog, QgsFillSymbol, QgsMarkerSymbol, QgsLineSymbol

CLASS_COLORS = {"Healthy": "#39FF14", "Yellowish": "#FFFF00", "Dead": "#000000"}


def load_detection_layers(iface, layers, plugin_name, class_colors=None):
    """Add bbox/centroid shapefiles and style them (red by default, or per-class colour)."""
    for info in layers:
        try:
            layer = iface.addVectorLayer(info["path"], info["name"], "ogr")
            if not layer:
                continue
            color = "red"
            if class_colors is not None:
                color = class_colors.get(info.get("class_name"), "red")
            if info["type"] == "bbox":
                symbol = QgsFillSymbol.createSimple(
                    {"color": "transparent", "outline_color": color, "outline_width": "1"})
            elif class_colors is not None:
                symbol = QgsMarkerSymbol.createSimple({"color": color, "size": "2", "outline_color": "white"})
            else:
                symbol = QgsMarkerSymbol.createSimple({"color": color, "size": "2"})
            layer.renderer().setSymbol(symbol)
            layer.triggerRepaint()
            iface.layerTreeView().refreshLayerSymbology(layer.id())
        except Exception as e:  # noqa
            QgsMessageLog.logMessage(f"Unable to load the output layer into QGIS: {e}", plugin_name, Qgis.Warning)


def report_failure(iface, error, canceled):
    """Shared handling for failed / canceled jobs. Returns True if it handled the case."""
    if canceled:
        iface.messageBar().pushMessage("Info", "The process was cancelled by the user.", level=Qgis.Info, duration=5)
        return True
    if error:
        iface.messageBar().pushMessage(
            "Error", f"Process failed: {error}. See the Log Messages Panel for details.",
            level=Qgis.Critical, duration=15)
        return True
    return False


def load_road_layers(iface, layers):
    for l in layers:
        if l["type"] == "raster":
            import os
            iface.addRasterLayer(l["path"], os.path.basename(l["path"]))
        else:
            import os
            layer = iface.addVectorLayer(l["path"], os.path.basename(l["path"]), "ogr")
            if layer:
                symbol = QgsLineSymbol.createSimple({"color": "red", "line_width": "1.5"})
                layer.renderer().setSymbol(symbol)
                layer.triggerRepaint()
                iface.layerTreeView().refreshLayerSymbology(layer.id())
