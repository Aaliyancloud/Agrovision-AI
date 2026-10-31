# AgroVision AI - Public + Exhibition package

This package keeps the supplied AgroVision one-page website and adds a real image-classification model. It has two modes using the **same FP32 ONNX model weights and preprocessing**.

## Public GitHub Pages mode
- Upload the website files (`index.html`, `js/`, and any existing `assets/`) to the existing GitHub Pages repository.
- The browser downloads the FP32 ONNX model from Hugging Face on the first scan and runs inference locally in the browser using ONNX Runtime Web.
- The selected image is not sent to an AgroVision server. A first-time model download is about 88 MB and needs internet. Some older or low-memory phones may run it slowly or fail to allocate enough memory.
- Later scans may use the browser cache, but browser cache is not guaranteed to remain forever.

## Exhibition laptop mode
- Install Python 3.11 or newer. On Windows, double-click `run_local.bat`.
- Wait for Python dependencies to install.
- Open `http://127.0.0.1:8000`.
- The local Python API downloads the same FP32 ONNX model on the first scan. Once the model is cached on the laptop, scanning can run without internet. Keep the project folder and Hugging Face cache intact.
- Use Chrome or Edge and allow camera access.

## Model and honest limitations
Model: `rodynaemad/plant-disease-dinov2-small`, file `onnx/model_with_features.onnx` (FP32, about 88 MB). Model card: https://huggingface.co/rodynaemad/plant-disease-dinov2-small

The model card reports 98.9% leaf-level accuracy on a held-out PlantVillage test set. This is **not** an accuracy guarantee for real farms. The model was trained on a closed set of 38 crop-condition labels and PlantVillage images are mostly single leaves on plain backgrounds. Real field images with shadows, multiple leaves, clutter or different cameras can perform substantially worse.

Important change from the previous broken version: AgroVision no longer applies the extra centroid/OOD threshold that was rejecting leaf photos. It uses the model's strongest class prediction and shows tomato classes. If another crop is the strongest prediction, it says “Not classified as tomato”. This is not a perfect non-leaf detector: a non-leaf photo may still be misclassified as a tomato class. Confidence is the model's softmax score, not verified real-world probability. Always confirm important crop decisions with an agriculture specialist.

## Included features
- Existing page layout, navigation, upload, live camera, English/Urdu toggle, Judge Mode, tutorial and exhibition guide retained.
- Tomato-only disease screening using DINOv2-small ONNX.
- Disease Library cards open a detail dialog. Reference photos are linked for selected conditions; other entries are clearly marked when no photo is available.
- Scan History stored locally in the browser (up to 20 records).
- Current weather lookup through Open-Meteo after the user taps the button and grants browser location permission; the original manual temperature/humidity risk sliders remain.
- Floating Plant Assistant with curated, bilingual guided responses. It is not a generative chatbot.
- Reduced decorative motion.

## Privacy
Public mode performs model inference in the user's browser. Local mode sends the image only to the Python service on the same laptop. The included API does not save uploaded images. Scan history stays in local browser storage. The live-weather button shares location coordinates with Open-Meteo only after permission is granted.

## Model attribution
Model and PlantVillage dataset are released under CC BY-SA 4.0. DINOv2 backbone is Apache 2.0. Author: Rodayna Emad Mohamed. See the model card for full details.
