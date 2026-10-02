# Deploying the Retina Screen demo to the cloud

The Streamlit app needs only `models/best_model.keras`, `models/model_metadata.json`,
the `src/` package and `app/`. Two free hosting options are described below.

## Option 1 - Hugging Face Spaces (recommended, free CPU, permanent URL)

1. Create a free account at https://huggingface.co and click **New Space**.
   * Space name: `retinascreen-dr` (any name)
   * SDK: **Streamlit**
   * Hardware: CPU basic (free)
2. Clone the empty Space and copy the repository into it:

   ```bash
   git clone https://huggingface.co/spaces/<your-hf-username>/retinascreen-dr
   cd retinascreen-dr
   # copy from this repository
   cp -r ../diabetic-retinopathy-stage-detection/{app,src,models,requirements.txt} .
   ```

3. Spaces need the entry point at the root and a small header in `README.md`:

   ```bash
   cp app/streamlit_app.py streamlit_app.py
   cp app/chatbot.py chatbot.py
   ```

   Create `README.md` in the Space with exactly this front-matter:

   ```yaml
   ---
   title: Retina Screen - Diabetic Retinopathy Stage Detection
   emoji: 👁️
   colorFrom: red
   colorTo: gray
   sdk: streamlit
   sdk_version: "1.38.0"
   app_file: streamlit_app.py
   pinned: false
   ---
   ```

   Because `streamlit_app.py` now lives at the root, change the two path lines near the top to:

   ```python
   ROOT = Path(__file__).resolve().parent
   sys.path.insert(0, str(ROOT / "src"))
   sys.path.insert(0, str(ROOT))
   ```

4. Model files over 10 MB must go through Git LFS on Hugging Face:

   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git add .
   git commit -m "Deploy Retina Screen"
   git push
   ```

5. The Space builds in ~3 minutes and is served at
   `https://huggingface.co/spaces/<your-hf-username>/retinascreen-dr`.
   Use `requirements.txt` from this repository but you may remove `jupyter`, `nbformat`
   and `kaggle` to speed up the build.

## Option 2 - Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io, sign in with GitHub, **New app**.
3. Repository: your repo, branch `main`, main file path: `app/streamlit_app.py`.
4. Deploy. The model file (~20 MB) is committed to the repository, so nothing else is needed.

## Local run (for the video)

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Sample images for the demonstration are in `app/assets/samples/` (two real APTOS test images
per grade, copied by the notebook; their true grade is in the file name).
