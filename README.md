# Diabetic Retinopathy Stage Detection with Transfer Learning

Computer Vision coursework - BSc (Hons) Computer Science, NIBM (Coventry University), batch BSCCOMP24.2P.

A complete, reproducible pipeline that grades colour fundus photographs into the five stages of
diabetic retinopathy (No DR, Mild, Moderate, Severe, Proliferative) using ImageNet-pretrained CNNs,
plus a Streamlit screening application with Grad-CAM explanations and a built-in assistant.

| | |
|---|---|
| **Dataset** | APTOS 2019 Blindness Detection, 224 px version - Kaggle `sovitrath/diabetic-retinopathy-224x224-2019-data` (3,662 images, CC0) |
| **Framework** | TensorFlow 2 / Keras 3, OpenCV, scikit-learn, Streamlit |
| **Main model** | EfficientNetB0 + Ben Graham preprocessing, two-phase transfer learning |
| **Notebook** | [`notebooks/DR_Stage_Detection_Colab.ipynb`](notebooks/DR_Stage_Detection_Colab.ipynb) - runs everything on a free Colab GPU |
| **Demo video** | *(link added in the report)* |

---

## 1. Repository layout

```
.
├── notebooks/DR_Stage_Detection_Colab.ipynb   end-to-end notebook (Colab GPU)
├── src/dr_detection/                          the reusable, commented package
│   ├── config.py          all hyper-parameters, class names, dataset constants
│   ├── preprocessing.py   border crop, resize, Ben Graham, CLAHE, unsharp, edge map
│   ├── data.py            dataset discovery, robust label mapping, stratified split, tf.data
│   ├── augmentation.py    on-the-fly augmentation, class weights, over-sampling
│   ├── model.py           EfficientNet / ResNet50V2 / MobileNetV2 / DenseNet121 + head
│   ├── train.py           two-phase training with callbacks, run_experiment()
│   ├── evaluate.py        accuracy, P/R/F1, QWK, referable-DR metrics, all figures
│   ├── gradcam.py         Grad-CAM explainability
│   ├── visualize.py       EDA / preprocessing / augmentation figures
│   └── utils.py           seeding, JSON I/O, timing, hardware info
├── scripts/
│   ├── download_data.py       Kaggle download + unzip
│   ├── make_figures.py        EDA figures (no GPU needed)
│   ├── train.py               train + evaluate one model from the CLI
│   ├── compare_experiments.py preprocessing x backbone grid -> comparison table
│   ├── predict.py             grade images from the CLI (+ Grad-CAM overlays)
│   └── export_results.py      zip results/ + models/ for the report
├── app/
│   ├── streamlit_app.py   RetinaScreen demo (single image, batch, chatbot, model card)
│   ├── chatbot.py         RetinaBot - rule-based assistant
│   └── assets/samples/    sample fundus images for the demo
├── models/                best_model.keras + model_metadata.json (created by training)
├── results/               one folder per run: metrics.json, history.csv, figures, ...
├── docs/                  report, video transcript, architecture diagram
└── requirements.txt
```

## 2. Quick start

### Option A - Google Colab (recommended, ~10 min for the main model)

1. Open `notebooks/DR_Stage_Detection_Colab.ipynb` in Colab, choose **Runtime -> T4 GPU**.
2. Set `REPO_URL` in the first cell to this repository.
3. Run all cells; upload your `kaggle.json` when prompted.
4. The last cell downloads `dr_results_bundle.zip` (results, model, sample images) - unzip it into the repository root.

### Option B - locally

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # needs ~/.kaggle/kaggle.json
python scripts/make_figures.py                  # EDA + preprocessing figures
python scripts/train.py --save-best             # main experiment (GPU recommended)
python scripts/compare_experiments.py           # optional: full experiment grid
streamlit run app/streamlit_app.py              # demo application
```

A 90-second CPU smoke test of the whole pipeline: `python scripts/train.py --quick-test`.

## 3. Pipeline

1. **Data** - folders are scanned into a DataFrame with a robust label map (the Kaggle folder is
   spelt `Proliferate_DR`; a naive map silently drops those 295 images). Labels are cross-checked
   against `train.csv`. Stratified 70 / 10 / 20 split with a fixed seed so every experiment shares
   the same test images.
2. **Preprocessing** - black-border crop -> resize 224 px -> Ben Graham Gaussian-blur subtraction
   (`4*img - 4*blur + 128`) with a circular mask. CLAHE, unsharp masking and raw are available
   for comparison. Images are preprocessed once and cached as `.npz`.
3. **Augmentation & balancing** - Keras preprocessing layers on the GPU (flips, rotation, zoom,
   shift, brightness, contrast), training split only. Inverse-frequency class weights in the loss;
   random over-sampling available with `--oversample`.
4. **Model** - ImageNet backbone (`include_top=False`) -> GAP -> Dropout -> Dense 512 -> Dropout ->
   Dense 256 -> Dense 5 softmax. Backbone-specific input normalisation is part of the graph, so
   the saved model accepts raw 0-255 images.
5. **Training** - phase 1: frozen backbone, Adam 1e-4; phase 2: top 30 layers un-frozen (BatchNorm
   frozen), Adam 1e-5. EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger.
6. **Evaluation** - accuracy, per-class / macro / weighted precision-recall-F1, quadratic weighted
   kappa, one-vs-rest ROC-AUC, confusion matrices, *referable-DR* (grade >= 2) sensitivity /
   specificity / AUC, within-one-grade accuracy, most-confident-error gallery, Grad-CAM grid.

## 4. Results

Numbers are produced by the notebook / scripts and stored in `results/<run>/metrics.json`;
`results/experiments_comparison.csv` collects every run. See the report for the discussion.

## 5. Demo application

`streamlit run app/streamlit_app.py`

* **Screen an image** - upload or pick a sample -> preprocessing preview, grade probabilities,
  Grad-CAM heat-map, referral recommendation (ICDR scale), inference time.
* **Batch screening** - grade a folder of images, download a CSV report.
* **RetinaBot** - offline, rule-based assistant that explains the result, the DR stages, the
  heat-map and the model's limitations (no API keys, no hallucinated medical advice).
* **Model card** - dataset, architecture, metrics, intended use, limitations, ethics.

Deployment files for Hugging Face Spaces are described in `docs/DEPLOYMENT.md`.

## 6. Reproducibility

Seeds are fixed (`Config.seed = 42`) for Python, NumPy and TensorFlow; the exact configuration
of every run is written to `results/<run>/config.json` together with the hardware used.

## 7. References

* Asia Pacific Tele-Ophthalmology Society (2019). *APTOS 2019 Blindness Detection*. Kaggle.
* Graham, B. (2015). *Kaggle Diabetic Retinopathy Detection competition report*. University of Warwick.
* Gulshan, V. et al. (2016). Development and validation of a deep learning algorithm for detection
  of diabetic retinopathy in retinal fundus photographs. *JAMA, 316*(22), 2402-2410.
* Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via
  gradient-based localization. *ICCV 2017*.
* Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural
  networks. *ICML 2019*.
* Wilkinson, C. P. et al. (2003). Proposed international clinical diabetic retinopathy and diabetic
  macular edema disease severity scales. *Ophthalmology, 110*(9), 1677-1682.

---
*This software is a university coursework prototype and is not a medical device.*
