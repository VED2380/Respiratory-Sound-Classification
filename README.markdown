# Respiratory Sound Classification Using ResNet Models

This project implements ResNet18 and ResNet50 models to classify respiratory sounds (Normal, Crackles, Wheezes, Both) from the ICBHI Respiratory Sound Database (6898 samples). It achieves ~80% accuracy using 2-fold interpatient cross-validation, focal loss, and class weights to handle imbalance (Normal: 3642, Both: 506). The project includes visualizations (architecture diagrams, spectrograms, confusion matrices, radar charts) to analyze performance.

## Project Structure
```
respiratory-sound-classification/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── visualization.py
├── docs/
│   ├── project_summary.md
├── scripts/
│   ├── preprocess_data.py
```

## Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for TensorFlow)
- Graphviz (for Figure 1)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/respiratory-sound-classification.git
   cd respiratory-sound-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Graphviz:
   - Download from [Graphviz](https://graphviz.org/download/) and add to PATH.
4. Prepare data:
   - Place `spectrograms_resized.npy`, `labels.npy`, and `patient_ids.npy` in `data/spectrograms/`.
   - See `data/README.md` for details.

## Usage
Run the main script to train models and generate figures:
```bash
python src/main.py
```
- **Output**: Figures 1–7 saved in `output/figures/`.
- **Training time**: ~1-1.5 hours on a GPU.

## Results
- **ResNet18**: ~80.67% accuracy
- **ResNet50**: ~81.60% accuracy
- See `docs/project_summary.md` for details.

## License
MIT License. See `LICENSE` for details.

## Acknowledgments
- ICBHI Respiratory Sound Database
- TensorFlow, Scikit-learn, Matplotlib, Seaborn, Graphviz
