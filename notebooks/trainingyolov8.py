# -*- coding: utf-8 -*-
"""TrainingYOLOv8.ipynb
"""

from google.colab import drive
drive.mount('/content/drive')

# !pip install ultralytics

# !pip install --upgrade sympy

# !pip install sympy==1.13.1

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  

# Train the model using augmentation parameters
results = model.train(
   data='/content/drive/MyDrive/YOLO-Tree Counting/Dataset/Dataset Estate Mapping/data.yaml',
   epochs=300,
   imgsz=640,
   augment=True,      
   hsv_h=0.020,       # Colour change (hue)
   hsv_s=0.8,         # Changes in colour saturation
   hsv_v=0.5,         # Changes in image brightness
   degrees=15.0,      # Rotate the image by up to 15 degrees
   translate=0.1,     # Move the image
   scale=0.6,         # Zoom in/out on the image
   flipud=0.5,        # There is a 50 per cent chance that the image is flipped vertically
   mixup=0.1          # Combining two images and their labels makes the model more general
)

# Save the model
model.save('/content/drive/MyDrive/YOLO-Tree Counting/Model/yolov8n_classification_2.1.pt')

from IPython.display import Image, display
import os

# Path to the training results directory
results_dir = '/content/drive/MyDrive/YOLO-Tree Counting/Model/Performance'

# List of plot files to display
plot_files = ['results_EM.png', 'confusion_matrix_EM.png', 'F1_curve_EM.png', 'P_curve_EM.png', 'R_curve_EM.png', 'PR_curve_EM.png']

# Display each plot
for plot_file in plot_files:
    plot_path = os.path.join(results_dir, plot_file)
    if os.path.exists(plot_path):
        print(f"Displaying {plot_file}:")
        display(Image(filename=plot_path))
    else:
        print(f"Plot file not found: {plot_file}")

from google.colab import runtime
print("Disconnecting runtime.")
runtime.unassign()