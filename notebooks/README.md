# **R&D Notebooks: PalmAI Development History**
## Welcome to the engine room of PalmAI. This directory contains the original research scripts, training pipelines, and experimental notebooks that power the QGIS plugin. This R&D approach bridges the gap between raw Deep Learning computer vision and practical Geospatial topology, ensuring that AI predictions translate into accurate, production-ready vector data.

---

## **Palm Oil Tree Analysis (YOLOv8)**
The first phase of research focused on object detection to automate tree counting, precise center-point extraction (for *Oryctes* pest spraying), and health classification.

### Key Research Highlights
- **Iterative Slicing:** Handling massive Orthophotos (GSD 2 cm to 7 cm) by slicing them into 640x640 and 1280x1280 pixel chips.
- **Data Balancing via Specific Slicing:** To prevent model overfitting on "Healthy" trees (which dominate the landscape), a script specifically targeted coordinates containing minority classes ("Yellowish" and "Dead") using ground-truth Shapefiles.
- **HSV Augmentation:** Fine-tuning Hue, Saturation, and Value (HSV) parameters during training to force the YOLOv8 model to recognize subtle color degradation indicating tree diseases.
- **Performance:** Achieved an mAP50 of **0.971** for tree counting.

---

## **Road Network Extraction (Semantic Segmentation)**
Extracting roads from plantation orthophotos presents a unique challenge: Canopy Occlusion. Palm fronds frequently cover the roads, causing standard segmentation models to predict broken, disconnected road segments.

### 1. **Model Evaluation**
To find the best foundational model, four different architectures were trained and evaluated against a manually digitized dataset of 1,200 images (3000x3000px, sliced down to 750px tiles):
- **U-Net (Scratch):** Custom-built architecture as a baseline.
- **U-Net (Pretrained):** Utilizing `segmentation_models_pytorch` with a ResNet50 backbone.
- **SegFormer:** A Vision Transformer (ViT) approach using HuggingFace's `nvidia/segformer-b0`.
- **DeepLabV3+:** Utilizing `torchvision` with a ResNet50 backbone and Atrous Spatial Pyramid Pooling. **(Winner)**

### 2. **Hybrid Post-Processing Pipeline**
To solve the canopy occlusion problem, a 4-stage post-processing pipeline was developed that acts on the model's output before vectorization:
- **Cleaner:** Applies a probability threshold, removes small pixel islands, and uses morphological Binary Opening to reduce noise.
- **Gap Filler:** Applies morphological Binary Closing to merge very close segments.
- **Smart Pathfinding (A* Algorithm):**
  - Skeletonizes the mask to find "dead ends" (endpoints).
  - Uses a KDTree to pair nearby endpoints within a maximum gap distance.
  - Inverts the AI's probability map to create a "Cost Map".
  - Runs the `skimage.graph.route_through_array` algorithm to find the cheapest path connecting endpoints, drawing a line through trees where a road is logically most likely to exist.
- **Topology Vectorization:** Converts the repaired skeleton into Shapely LineStrings and filters out micro-segments using GeoPandas.

### 3. **Hyperparameter Tuning (Grid Search)**
Because the Hybrid Pipeline relies on 10 different parameters, an automated Grid Search script determined the optimal combination, calculating Raster IoU for cleaning and Vector IoU (with a 3-meter buffer) for the final shapefile output.

#### **Understanding Post-Processing Parameters:**
**A. Cleaner Parameters**
1. `PROBABILITY_THRESHOLD`: Confidence cutoff (0.0 to 1.0) converting sigmoid output into a binary road mask.
2. `MIN_OBJECT_SIZE_PIXELS`: Removes isolated small pixel islands (noise/false positives) smaller than this area.
3. `OPENING_KERNEL_SIZE`: Morphological Binary Opening kernel size to smooth boundaries and remove salt-and-pepper noise.

**B. Gap Filler Parameter**
4. `CLOSING_ITERATIONS`: Iterations for morphological Binary Closing; bridges segments separated by a few pixels.

**C. Pathfinding Parameters**
5. `PATHFINDING_MAX_GAP_PIXELS`: Search radius (via KDTree) to connect dead ends.
6. `PATHFINDING_COST_FACTOR`: Multiplier (e.g., 15) used to invert probability into a traversable Cost Map.
7. `PATHFINDING_MAX_AVG_COST`: Maximum allowed average cost for an A* path, preventing lines through dense non-road areas.
8. `PATHFINDING_BBOX_PADDING`: Optimization padding (in pixels) around endpoint bounding boxes to restrict search space.

**D. Vector Filter Parameters**
9. `MIN_ROAD_LENGTH_METERS`: Drops LineStrings shorter than this length (in meters) to clean micro-segments.
10. `SIMPLIFICATION_TOLERANCE_METERS` (Deprecated): Disabled in final release to preserve raw topological skeletons.

#### **Final Tuning Results:**
| Parameter | U-Net (Scratch) | U-Net (Pretrained) | SegFormer | DeepLabV3+ (Best) |
|---|:---:|:---:|:---:|:---:|
| Score (IoU) | `0.6579` | `0.6808` | `0.6679` | **`0.6935`** |
| `OPENING_KERNEL_SIZE` | 9 | 5 | 3 | **2** |
| `MIN_OBJECT_SIZE_PIXELS` | 771 | 999 | 909 | **551** |
| `CLOSING_ITERATIONS` | 50 | 50 | 50 | **50** |
| `PROBABILITY_THRESHOLD` | 0.2970 | 0.5035 | 0.4058 | **0.4316** |
| `PATHFINDING_MAX_GAP_PIXELS` | 300 | 350 | 550 | **600** |
| `PATHFINDING_AVG_COST` | 1.063 | 0.7385 | 0.6919 | **1.1164** |
| `BBOX_PADDING` | 40 | 25 | 35 | **25** |
| `MIN_ROAD_LENGTH` | 14.0m | 14.0m | 16.0m | **14.0m** |

*(Note: `PATHFINDING_COST_FACTOR` was locked at 15 for all runs).*

---

## **Bridging Research to Production Deployment**

The algorithms developed and tuned in these notebooks are deployed directly into PalmAI's production engine:
1. **Model Weights Release (`models-v1`):**  
   Trained checkpoints (`yolov8n_counting_1.0.pt`, `yolov8n_centertree_1.3.pt`, `yolov8n_classification_2.1.pt`, `deeplabv3+_roaddetection_1.pth.tar`) are published as GitHub Release assets and guarded by cryptographic SHA-256 verification in `plugin_qgis/palmai_core/model_store.py`.
2. **Production Pipelines:**  
   `plugin_qgis/palmai_core/yolo.py` and `plugin_qgis/palmai_core/road.py` run these exact tuned parameters in containerized or isolated virtualenv backends.
3. **Execution Provenance:**  
   Every inference run produces a `run_manifest.json` recording model checksums, runtime versions, CRS, and output verification hashes.

---

## **Repository Manifest**
Below is the index of core R&D scripts preserved in this directory:

### **YOLOv8 Pipeline (Trees)**
- `slicing_to_png_randomize.py` - Standard tile generation.
- `spesific_slicing_to_png.py` - Targeted slicing for class balancing.
- `trainingyolov8.py` - Model training with HSV augmentation.

### **Road Segmentation Pipeline (Roads)**
**Training Scripts:**
- `u_net_scratch_training.py`
- `u_net_pretrained_training.py`
- `deeplabv3+_training.py`
- `segformer_training.py`

**Inference & Hybrid Post-Processing Scripts:**
- `u_net_scratch_detection_hybrid_postprosessing.py`
- `u_net_pretrained_detection_hybrid_postprocessing.py`
- `deeplabv3+_detection_hybrid_postprosessing.py`
- `segformer_detection_hybrid_postprosessing.py`
- `post_processing_parameter_tuning_(gridsearch).py` - Automated script that produced the parameter optimization matrix above.
