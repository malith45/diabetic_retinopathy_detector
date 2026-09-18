"""
dr_detection
============

A small, modular Python package for **Diabetic Retinopathy (DR) stage
detection** from colour fundus photographs using transfer learning.

The package is organised so that every step of the pipeline described in the
coursework brief lives in its own module:

* ``dr_detection.config``        - all tunable settings in one place
* ``dr_detection.preprocessing`` - border cropping, Ben Graham, CLAHE, edge enhancement
* ``dr_detection.data``          - dataset discovery, label mapping, stratified splits, tf.data
* ``dr_detection.augmentation``  - on-the-fly augmentation + class balancing helpers
* ``dr_detection.model``         - CNN backbones with transfer-learning heads
* ``dr_detection.train``         - two-phase training strategy with callbacks
* ``dr_detection.evaluate``      - metrics, confusion matrix, curves, error analysis
* ``dr_detection.gradcam``       - Grad-CAM explainability
* ``dr_detection.visualize``     - exploratory data analysis figures
* ``dr_detection.utils``         - seeding, I/O and timing helpers

The same code is used by the Colab notebook, the command-line scripts and the
Streamlit demo application, so results are reproducible everywhere.
"""

__version__ = "1.0.0"
__author__ = "BSCCOMP24.2P - Computer Vision Coursework"
