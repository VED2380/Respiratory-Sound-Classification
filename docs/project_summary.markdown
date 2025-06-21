# Respiratory Sound Classification Project Summary

**Jan 2025 - June 2025**

- Developed ResNet18 and ResNet50 models to classify respiratory sounds (Normal, Crackles, Wheezes, Both) from the ICBHI Respiratory Sound Database (6898 samples), achieving \~80% accuracy with 2-fold interpatient cross-validation.
- Utilized preprocessed STFT spectrograms (75x50), ensuring 6898 samples and resolving prior data mismatch (1403 vs. 6898).
- Applied corrected focal loss and class weights to address imbalance (Normal: 3642, Both: 506), optimizing with 50 epochs and Adam.
- Created comprehensive visualizations (architecture diagrams, spectrogram examples, annotated confusion matrices, radar charts) to analyze performance.
- Leveraged Python, TensorFlow, Scikit-learn, Matplotlib, Seaborn, and Graphviz, showcasing expertise in deep learning, audio processing, and visualization.
- **Impact**: Advanced automated respiratory diagnosis with insights for clinical deployment.