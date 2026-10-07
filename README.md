# ECG Heartbeat Classification with a 1D CNN

Classifying ECG heartbeats from the MIT-BIH Arrhythmia dataset into five beat types, with exploratory analysis, class balancing and a 1D convolutional neural network.

> Learning project completed during my bachelor's degree. Not intended for clinical use.

## Highlights

- Class distribution analysis and visualisation
- Upsampling of minority classes to balance the training set
- Sample ECG plots and per-class intensity heatmaps
- A 1D CNN trained and evaluated on the held-out test file

## Dataset

The preprocessed MIT-BIH Arrhythmia data (`mitbih_train.csv`, `mitbih_test.csv`), available from the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) on Kaggle and originally from PhysioNet. The CSV files are not included in this repository; the notebook downloads them automatically with `kagglehub`.

Each row is one heartbeat followed by its class label:

| Label | Beat type |
|---|---|
| 0 | Non-ectopic (N) |
| 1 | Supraventricular ectopic (S) |
| 2 | Ventricular ectopic (V) |
| 3 | Fusion (F) |
| 4 | Unknown (Q) |

## Method

- **Balancing:** the training set is heavily imbalanced, so each class is resampled to 20,000 beats (100,000 training samples in total).
- **Model:** three Conv1D layers (64 filters each) with batch normalisation and max pooling, followed by dense layers of 64 and 32 units and a 5-way softmax output.
- **Training:** Adam optimiser, categorical cross-entropy, 5 epochs, batch size 32, with 10% of the training data held out for validation.

## Results

Test accuracy from the latest run: **96.01%** on the held-out MIT-BIH test file (21,892 beats). Validation uses a 10% split of the training data, so the test set is not seen during training. The notebook also contains the confusion matrix and training curves, and `RESULTS.md` records the per-epoch figures.

One caveat: the upsampling happens before the validation split, so validation accuracy is optimistic. The test accuracy is the figure to rely on.

## Running it

```bash
pip install "tensorflow-cpu==2.15.*" "numpy<2" pandas scikit-learn matplotlib seaborn kagglehub notebook
jupyter notebook MED_ECG_Classification_with_cnn.ipynb
```

The notebook downloads the dataset from Kaggle on first run. It was last run end to end with Python 3.10 and TensorFlow 2.15 by the GitHub Actions workflow in `.github/workflows/rerun.yml`, which can be started manually from the Actions tab.

## Next steps

- Split before upsampling so the validation data contains no duplicated beats
- Try LSTM or hybrid CNN-LSTM models
- Evaluate with a patient-wise split

## License

MIT. See [LICENSE](LICENSE).
