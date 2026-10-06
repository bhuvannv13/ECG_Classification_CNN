# ECG Heartbeat Classification with a 1D CNN

Classifying ECG heartbeats from the MIT-BIH Arrhythmia dataset into five beat types, with exploratory analysis, class balancing and a 1D convolutional neural network.

> Learning project completed during my bachelor's degree. Not intended for clinical use.

## Highlights

- Class distribution analysis and visualisation
- Upsampling of minority classes to balance the training set
- Sample ECG plots and per-class intensity heatmaps
- A 1D CNN trained and evaluated on the held-out test file

## Dataset

The preprocessed MIT-BIH Arrhythmia data (`mitbih_train.csv`, `mitbih_test.csv`), available from the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) on Kaggle and originally from PhysioNet. The CSV files are not included in this repository.

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
- **Training:** Adam optimiser, categorical cross-entropy, 5 epochs, batch size 32.

## Results

Test accuracy recorded in the notebook: **96.91%**. The notebook also contains the confusion matrix and training curves.

One caveat: the test file is also used as validation data during training, so this figure is slightly optimistic. Holding out part of the training set for validation would give a cleaner estimate.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow notebook
jupyter notebook MED_ECG_Classification_with_cnn.ipynb
```

The notebook was written in Google Colab. Update the two `pd.read_csv` paths to where you saved the dataset.

## Next steps

- Use a separate validation split
- Try LSTM or hybrid CNN-LSTM models
- Evaluate with a patient-wise split

## License

MIT. See [LICENSE](LICENSE).
