# **R&D Notebooks: PalmAI Development History**
## Welcome to the engine room of PalmAI. This directory contains the original research scripts, training pipelines, and experimental notebooks that power the QGIS plugin. This R&D approach bridges the gap between raw Deep Learning computer vision and practical Geospatial topology, ensuring that AI predictions translate into accurate, production-ready vector data.

## **Palm Oil Tree Analysis (YOLOv8)**
The first phase of my research focused on object detection to automate tree counting, precise center-point extraction (for Oryctes pest spraying), and health classification.
### Key Reseacrh Highlight
- **Iterative Slicing:** Handling massive Orthophotos (GSD 2cm 7cm) by slicing them into 640x640 and 1280x1280 pixel chips.
- **Data Balancing via Specific Slicing:** To prevent model overfitting on "Healthy" trees (which dominate the landscape), I developed a script to specifically target and slice coordinates containing the minority classes ("Yellowish" and "Dead") using ground-truth Shapefiles.
- **HSV Augmentation:** Fine-tuning the Hue, Saturation, and Value (HSV) parameters during training to force the YOLOv8 model to recognize the subtle color degradation indicating tree diseases.
**Performance:** Achieved an mAP50 of 0.971 for tree counting.

## **Road Network Extraction (Semantic Segmentation)**
Extracting roads from plantation orthophotos presents a unique challenge: Canopy Occlusion. Palm fronds frequently cover the roads, causing standard segmentation models to predict broken, disconnected road segments.

### 1. **Model Evaluation**
To find the best foundational model, I trained and evaluated four different architectures against a manually digitized dataset of 1,200 images (3000x3000px, sliced down to 750px tiles):
-  **U-Net (Scratch):** Custom-built architecture as a baseline.
-  **U-Net (Pretrained):** Utilizing `segmentation_models_pytorch` with a ResNet50 backbone.
-  **SegFormer:** A Vision Transformer (ViT) approach using HuggingFace's `nvidia/segformer-b0`.
-  **DeepLabV3+:** Utilizing `torchvision` with a ResNet50 backbone and Atrous Spatial Pyramid Pooling. **(Winner)**

### 2. **Hybrid Post-Processing Pipeline**
To solve the canopy occlusion problem, i developed a 4-stage post-processing pipeline that acts on the model's output before vectorization:
- **Cleaner:** Applies a probability threshold, removes small pixel islands, and uses morphological Binary Opening to reduce noise.
- **Gap Filler:** Applies morphological Binary Closing to merge very close segments.
-  **Smart Pathfinding (A star Algorithm):**
    - Skeletonizes the mask to find "dead ends" (endpoints).
    - Uses a KDTree to pair nearby endpoints within a maximum gap distance.
    - Inverts the AI's probability map to create a "Cost Map".
    - Runs the `skimage.graph.route_through_array` algorithm to find the cheapest path connecting the endpoints, effectively drawing a line through the trees where a road is logically most likely to exist.
- **Topology Vectorization:** Converts the repaired skeleton into Shapely LineStrings and filters out micro-segments using GeoPandas.

### 3. **Hyperparameter Tuning (Grid Search)**
Because the Hybrid Pipeline relies on 10 different parameters, i built a Grid Search script to automatically find the optimal combination. It calculates the Raster IoU for the cleaning phases and the Vector IoU (with a 3-meter buffer) for the final shapefile output.

#### **Understanding the Post-Processing Parameters:**
**A. Cleaner Parameters**
1. `PROBABILITY_THRESHOLD`: The confidence cutoff (0.0 to 1.0) to convert the AI's continuous sigmoid output into a binary road/non-road mask.
2. `MIN_OBJECT_SIZE_PIXELS`: Removes isolated, small pixel islands (noise/false positives) that have an area smaller than this value.
3. `OPENING_KERNEL_SIZE`: The size of the kernel used for morphological Binary Opening. It smooths boundaries and removes salt-and-pepper noise.

**B. Gap Filler Parameter**

4.  `CLOSING_ITERATIONS`: The number of iterations for morphological Binary Closing. Acts as a brute-force "glue" to connect road segments that are only separated by a few pixels.

**C. Pathfinding Parameters**

5. `PATHFINDING_MAX_GAP_PIXELS`: The maximum search radius (via KDTree) the algorithm will look to connect two dead ends. Gaps larger than this are ignored to save memory.
6. `PATHFINDING_COST_FACTOR`: A locked multiplier (e.g., 15) used to invert the probability map into a traversable Cost Map.
7. `PATHFINDING_MAX_AVG_COST`: The maximum allowed average cost for an A* path. This is a critical filter that prevents the algorithm from forcefully drawing roads through dense, highly improbable non-road areas.
8. `PATHFINDING_BBOX_PADDING`: An optimization parameter that adds a buffer (in pixels) around the bounding box of two endpoints, restricting the A* search area so it doesn't process the entire massive map.

**D. Vector Filter Parameters**

9. `MIN_ROAD_LENGTH_METERS`: Operates on the final vector geometries. It drops any generated LineString that is shorter than this length (in meters), cleaning up useless micro-segments.
10. `SIMPLIFICATION_TOLERANCE_METERS` (Deprecated): Originally used for Douglas-Peucker line simplification, but was disabled in the final version to preserve the raw, precise topological skeleton.

#### **Final Tuning Results:**
| Parameter | U-Net (Scratch) | U-Net (Pretrained) | SegFormer |DeepLabV3+ (Best) |
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

(Note: PATHFINDING_COST_FACTOR was locked at 15 for all runs).

## **Repository Manifest**
Below is the index of the core R&D scripts preserved in this directory:

### **YOLOv8 Pipeline (Trees)**
- `slicing_to_png_randomize.py` - Standard tile generation.
- `spesific_slicing_to_png.py` )- Targeted slicing for class balancing.
- `trainingyolov8.py` - Model training with HSV augmentation.

### **Road Segmentation Pipeline (Roads)**
### Training Scripts:
- `u_net_scratch_training.py`
- `u_net_pretrained_training.py`
- `deeplabv3+_training.py`
- `segformer_training.py`
### Inference & Hybrid Post-Processing Scripts:
- `u_net_scratch_detection_hybrid_postprosessing.py`
- `u_net_pretrained_detection_hybrid_postprocessing.py`
- `deeplabv3+_detection_hybrid_postprosessing.py`
- `segformer_detection_hybrid_postprosessing.py`
### Inference & Hybrid Post-Processing Scripts:
- `post_processing_parameter_tuning_(gridsearch).py` - The automated script used to generate the parameter table above.
