# Diabetic Retinopathy Stage Detection with Transfer Learning and External Validation

Computer Vision coursework - BSc (Hons) Computing, NIBM (Coventry University), batch BSCCOMP24.2P.034

A reproducible pipeline that grades colour fundus photographs into the five stages of diabetic
retinopathy (ICDR scale: No DR, Mild, Moderate, Severe, Proliferative) with an ImageNet-pretrained CNN,
validates it on an independent dataset from another country, and serves it in a Streamlit screening
application with Grad-CAM explanations and a built-in assistant.

| | |
|---|---|
| **Development data** | APTOS 2019, 224 px version - Kaggle `sovitrath/diabetic-retinopathy-224x224-2019-data` (Version 4, 3,662 images, CC0) |
| **External test data** | EyePACS 2015, 224 px version - Kaggle `sovitrath/diabetic-retinopathy-2015-data-colored-resized` (35,126 images, CC0), never used for training or tuning |
| **Final model** | EfficientNetB3 (ImageNet), two-phase fine-tuning, class weights, test-time augmentation |
| **Main notebook** | `notebooks/` - the executed Kaggle notebook (Steps 1-11, GPU T4) |
| **Framework** | TensorFlow 2.20 / Keras 3, OpenCV, scikit-learn, Streamlit |
| **Links** | Live app: https://diabeticretinopathydetector034.streamlit.app · video and Kaggle notebook: see the report |

## Results

| Metric | APTOS 2019 test (internal, n = 523) | EyePACS 2015 (external, n = 35,126) |
|---|---|---|
| Accuracy | 0.809 | 0.688 |
| Macro-F1 | 0.645 | 0.314 |
| Quadratic weighted kappa | 0.900 | 0.388 |
| Referable-DR sensitivity (grade >= 2) | 0.847 | 0.310 |
| Referable-DR specificity | 0.963 | 0.945 |
| Referable-DR AUC | 0.979 | 0.745 |
| Within one grade | 0.962 | 0.833 |

* **13 controlled experiments** (about 140 GPU minutes) chose the preprocessing, backbone, class-balancing
  strategy and hyper-parameters on the validation set only; the test set was used once
  (`results/experiments_summary.csv`, `figures/09_experiment_results.png`).
* **Leakage audit**: a retinal-vessel similarity search found 148 duplicate photographs in APTOS, 43 with
  conflicting grades; 176 images were removed before the stratified 70 / 15 / 15 split.
* **External validation** shows a large drop caused by domain shift: the model under-grades EyePACS eyes.
  A site-specific threshold restores 81 % sensitivity only at 49 % specificity
  (`results/external_threshold_calibration.json`), so local retraining would be required for deployment.

All figures are in `figures/`, all numbers in `results/`, and the full discussion is in the report.

## Repository layout

```
.
├── notebooks/                    Kaggle notebook (final, executed) and earlier development notebooks
├── figures/                      every figure produced by the Kaggle notebook (Steps 3-11)
├── results/                      metrics, histories and tables of every step and experiment
│   └── experiments/<run>/        history.csv, curves.png and metrics.json of each training run
├── models/
│   ├── best_model.keras          final EfficientNetB3 classifier (raw 0-255 input, normalisation inside)
│   └── model_metadata.json       preprocessing, TTA, metrics, datasets, environment
├── app/
│   ├── streamlit_app.py          Retina Screen prototype (screening, batch, RetinaBot, model card)
│   ├── chatbot.py                RetinaBot - offline, rule-based assistant
│   └── assets/samples/           real APTOS test images and EyePACS images (true grade in the file name)
├── src/dr_detection/
│   ├── kaggle_pipeline.py        the exact preprocessing used by the final model (shared with the app)
│   └── ...                       reusable package for command-line experiments (scripts/)
├── scripts/                      command-line training, prediction and figure scripts
├── docs/DEPLOYMENT.md            cloud hosting instructions (Streamlit Community Cloud, Hugging Face)
└── requirements.txt
```

## Run the prototype

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. Pick a bundled sample image or upload a fundus photograph to see
the predicted grade, the probability of every grade, the Grad-CAM heat-map, the referral advice, and the
preprocessing stages. The model and its metadata are already included, so no training is needed.

## Reproduce the training

1. Open the notebook in `notebooks/` on Kaggle, attach both datasets above (**Add Input**), and set
   **Accelerator: GPU T4 x2** and **Internet: On** (needed for the ImageNet weights).
2. Run all cells. Steps 1-8 take about 10 minutes, the 13 experiments of Step 9 about 2.5 hours, and the
   evaluation and external validation (Steps 10-11) about 20 minutes.
3. Download `dr_results_bundle.zip` from the notebook output and unzip it into this repository.

All random generators are seeded (`SEED = 42`), the split is stored in `results/split_aptos2019.csv`, and
the configuration of every run is stored with its results.

## Pipeline summary

1. **Data** - folder labels mapped to ICDR grades (including the `Proliferate_DR` spelling), cross-checked
   with the official CSV files; duplicate audit; stratified 70 / 15 / 15 split.
2. **Preprocessing** - crop black border, pad to square, resize 224 px, retina mask. CLAHE, a measured noise
   filter, unsharp masking and Ben Graham's method were evaluated; after fine-tuning, no extra filtering
   performed best.
3. **Augmentation and balancing** - on-the-fly flips, rotation, zoom, shift, brightness and contrast (training
   only); class weights, balanced sampling and no balancing compared.
4. **Model** - EfficientNetB0/B3, ResNet50V2, DenseNet121 compared; head GAP -> Dense 512 -> Dense 256 ->
   softmax(5); phase 1 frozen backbone (Adam 1e-3), phase 2 full fine-tuning with BatchNorm frozen (Adam 1e-4).
5. **Training strategy** - early stopping on validation QWK, ReduceLROnPlateau, model checkpointing.
6. **Evaluation** - classification report, confusion matrices, ROC, QWK, referable-DR sensitivity and
   specificity, error analysis, calibration (ECE), Grad-CAM, external validation, site calibration.

## Key references

* Asia Pacific Tele-Ophthalmology Society (2019). *APTOS 2019 Blindness Detection* [Data set]. Kaggle.
* Gulshan, V. et al. (2016). Development and validation of a deep learning algorithm for detection of
  diabetic retinopathy in retinal fundus photographs. *JAMA, 316*(22), 2402-2410.
* Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based
  localization. *ICCV 2017*.
* Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
  *ICML 2019*.
* Wilkinson, C. P. et al. (2003). Proposed international clinical diabetic retinopathy and diabetic macular
  edema disease severity scales. *Ophthalmology, 110*(9), 1677-1682.

---
*This software is a university coursework prototype and is not a medical device.*
