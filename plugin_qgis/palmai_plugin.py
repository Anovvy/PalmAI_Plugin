# -*- coding: utf-8 -*-
"""
/***************************************************************************
 PalmAI
                                 A QGIS plugin
 Mapping automation plugin for Palm Oil Plantations
                              -------------------
        begin                : 2026
        copyright            : (C) 2026 by Bayu Nabiil
        email                : bayunabiil1365@gmail.com
 ***************************************************************************/
"""

import os.path
from qgis.core import QgsApplication

# --- Set the environment variables for GDAL/PROJ ---
gdal_driver_path = os.path.join(QgsApplication.prefixPath(), 'bin', 'gdalplugins')
if 'GDAL_DRIVER_PATH' not in os.environ:
    os.environ['GDAL_DRIVER_PATH'] = gdal_driver_path

proj_lib_path = os.path.join(QgsApplication.prefixPath(), 'share', 'proj')
if 'PROJ_LIB' not in os.environ:
    os.environ['PROJ_LIB'] = proj_lib_path
# --- END OF CODE BLOCK ---

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu

# =========================================================================
# --- LAZY LOADING ---
# The import dialogue has been moved to the 'run_...' function to speed up QGIS startup
# =========================================================================

class PalmAIPlugin:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir, 'i18n', f'PalmAIPlugin_{locale}.qm')

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = self.tr(u'&PalmAI')
        
        # Instance dialog (‘None’ at the start for lazy loading)
        self.counting_dlg = None
        self.classification_dlg = None
        self.centertree_dlg = None
        self.road_detection_dlg = None

    def tr(self, message):
        """Get the translation for a string using Qt translation API."""
        return QCoreApplication.translate('PalmAIPlugin', message)

    def initGui(self):
        """Create the menu entries for the plugin."""
        plugins_menu = self.iface.pluginMenu()
        
        # --- SETUP MAIN MENU ---
        self.plugin_menu = QMenu(self.tr("PalmAI"), plugins_menu)
        main_icon_path = os.path.join(self.plugin_dir, 'icons', 'palmai_icon.jpg')
        if os.path.exists(main_icon_path):
            self.plugin_menu.setIcon(QIcon(main_icon_path))

        plugins_menu.addMenu(self.plugin_menu)
        self.actions = []
    
        #--- Action for "Palm Oil Tree Counting" ---
        counting_icon_path = os.path.join(self.plugin_dir, 'icons', 'counting_icon.png')
        counting_action = QAction(
            QIcon(counting_icon_path),
            self.tr("Palm Oil Tree Counting"), 
            self.iface.mainWindow()
        )
        counting_action.triggered.connect(self.run_counting_tool)
        self.actions.append(counting_action)
        self.plugin_menu.addAction(counting_action)

        # --- Action for "Palm Oil Center Tree" ---
        centertree_icon_path = os.path.join(self.plugin_dir, 'icons', 'centertree_icon.png')
        center_tree_action = QAction(
            QIcon(centertree_icon_path),
            self.tr("Palm Oil Center Tree"), 
            self.iface.mainWindow()
        )
        center_tree_action.triggered.connect(self.run_center_tree_tool)
        self.actions.append(center_tree_action)
        self.plugin_menu.addAction(center_tree_action)
        
        # --- Action for "Palm Oil Tree Classification" ---
        classification_icon_path = os.path.join(self.plugin_dir, 'icons', 'classification_icon.png')
        classification_action = QAction(
            QIcon(classification_icon_path),
            self.tr("Palm Oil Tree Classification"), 
            self.iface.mainWindow()
        )
        classification_action.triggered.connect(self.run_classification_tool)
        self.actions.append(classification_action)
        self.plugin_menu.addAction(classification_action)

        # --- Action for "Road Detection" ---
        road_icon_path = os.path.join(self.plugin_dir, 'icons', 'road_detection_icon.png') 
        road_action = QAction(
            QIcon(road_icon_path),
            self.tr("Road Detection (DeepLab)"), 
            self.iface.mainWindow()
        )
        road_action.triggered.connect(self.run_road_detection_tool)
        self.actions.append(road_action)
        self.plugin_menu.addAction(road_action)

    def unload(self):
        """Removes the plugin menu from the QGIS GUI."""
        self.iface.pluginMenu().removeAction(self.plugin_menu.menuAction())

    # --- Functions to run each dialog ---
    def run_counting_tool(self):
        from .scripts.counting_dialog import CountingDialog
        if self.counting_dlg is None:
            self.counting_dlg = CountingDialog(self.iface, self.iface.mainWindow())
        self.counting_dlg.show()
        self.counting_dlg.activateWindow()
    
    def run_center_tree_tool(self):
        from .scripts.centertree_dialog import CenterTreeDialog
        if self.centertree_dlg is None:
            self.centertree_dlg = CenterTreeDialog(self.iface, self.iface.mainWindow())
        self.centertree_dlg.show()
        self.centertree_dlg.activateWindow()

    def run_classification_tool(self):
        from .scripts.classification_dialog import ClassificationDialog
        if self.classification_dlg is None:
            self.classification_dlg = ClassificationDialog(self.iface, self.iface.mainWindow())
        self.classification_dlg.show()
        self.classification_dlg.activateWindow()

    def run_road_detection_tool(self):
        from .scripts.road_detection_dialog import RoadDetectionDialog
        if self.road_detection_dlg is None:
            self.road_detection_dlg = RoadDetectionDialog(self.iface, self.iface.mainWindow())
        self.road_detection_dlg.show()
        self.road_detection_dlg.activateWindow()