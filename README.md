# Photo AI Organizer

A simple tool to scan a folder of photos and copy those similar to a given example image into a destination folder. It uses a pretrained CNN (ResNet50 via `timm`) and cosine similarity between feature embeddings.

Now includes a graphical interface (GUI) to pick folders/files and visualize each processed image. A green advisory appears when the current image is a match.

## Features
- Choose Source folder, Example image, and Destination folder.
- Set similarity threshold.
- Live preview of the currently processed image.
- Green "MATCH" indicator when an image exceeds the threshold and is copied to the destination.

## Requirements
- Python 3.8+
- PyTorch and torchvision
- timm
- scikit-learn
- Pillow

Install dependencies (example):

```bash
pip install torch torchvision timm scikit-learn pillow
```

Note: Installing PyTorch may require following the official installation instructions for your OS/CUDA.

## Usage

Run the application:

```bash
python main.py
```

1. Click “Scegli...” to select:
   - Cartella Sorgente (Source folder with images to scan)
   - Immagine Esempio (Example photo of the subject)
   - Cartella Destinazione (Where similar images will be copied)
2. Optionally adjust the similarity threshold (0–1). Suggested range: 0.85–0.95.
3. Click “Avvia Analisi”.
4. The main image panel will show each image being processed. When an image matches, a green "MATCH" banner will appear and the file will be copied to the destination folder.

The app will load the pretrained model on first run which may take some time.

## CLI (optional)
The original CLI function is preserved internally for reference, but the GUI starts by default when running `main.py`.

## License
MIT License. See LICENSE file.
