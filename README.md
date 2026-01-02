# Smart Waste Classifier

Real-time computer vision app that classifies waste from a webcam feed and recommends the correct disposal bin using a trained image classification model.

## Overview

This project uses OpenCV and cvzone to capture live video, run inference with a Keras model, and render predictions inside a custom UI background. Based on the predicted class, the app maps the item to a disposal category and overlays the corresponding bin and waste visuals.

## Demo

_In progress_

## How it works

1. Capture frames from the webcam (OpenCV).
2. Run classification using the trained Keras model.
3. Use `labels.txt` to map the predicted index to a class name.
4. Map the predicted class to a disposal bin category.
5. Overlay UI elements (background, arrow, bin image, waste icon) for clear feedback.

## Key features

- Live webcam classification using a trained Keras model
- On-screen UI overlay (custom background + visual feedback)
- Bin recommendation based on predicted class
- Modular assets for bins, waste icons, labels, and model files

## Tech stack

- Python
- OpenCV
- cvzone
- NumPy
- Keras/TensorFlow (inference)

## Model

- Model file: `Resources/Model/keras_model.h5`
- Labels: `Resources/Model/labels.txt`
- Notes: model trained using Teachable Machine and exported for Keras inference

## Classes and bin mapping

The model predicts one of the following classes (from `labels.txt`):

- Zip-top cans
- Newspaper
- Old shoes
- Watercolor pen
- Disinfectant
- Battery
- Vegetable leaf
- Apple

Classes are mapped to the correct disposal bin using `classDic` in `main.py`:

- Recyclable Waste (blue): Zip-top cans, Newspaper
- Hazardous Waste (red): Disinfectant, Battery
- Food Waste (green): Vegetable leaf, Apple
- Residual Waste (gray): Old shoes, Watercolor pen

## Project structure

```text
.
├── main.py
├── test_load.py
└── Resources
    ├── arrow.png
    ├── background.png
    ├── Bins
    ├── Waste
    └── Model
        ├── keras_model.h5
        ├── labels.txt
        └── converted_savedmodel
```

## Notes

- Requires a working webcam.
- Classification quality depends on lighting and how clearly the item is shown to the camera.
- If you change labels or add classes, update the class-to-bin mapping logic accordingly.

## Results

- Classes: 8 waste categories mapped to 4 bin types
- Accuracy: ~92% under ideal lighting; ~85–90% in typical conditions
- Performance: 15–20 FPS with predictions appearing within ~0.5 seconds
- Common failure cases: low lighting, transparent objects, and busy backgrounds

## Future improvements

- Add confidence thresholding and “unknown” handling
- Expand classes and retrain with more diverse images
- Smooth predictions across frames to reduce flicker
- Add a short demo GIF and screenshots for the README
