# Cat and dog facial feature classification

A project for data engineering and AI at AUT. I built a deep learning pipeline in PyTorch to classify cat and dog eyes, faces, mouths and noses, then compared a custom CNN with transfer learning and fine tuning for a Kaggle competition.

<br>

<p align="center">
  <a href="outputs/figures/reference_cnn_architecture.png"><img src="outputs/figures/reference_cnn_architecture.png" height="190" alt="Reference CNN architecture with convolution, pooling and dropout layers"></a>
  &nbsp;&nbsp;
  <a href="outputs/figures/E2_mobilenetv2_loss_accuracy_curves.png"><img src="outputs/figures/E2_mobilenetv2_loss_accuracy_curves.png" height="190" alt="Training and validation loss and accuracy during MobileNetV2 fine tuning"></a>
</p>
<p align="center"><em>Custom CNN architecture (left) and MobileNetV2 fine tuning (right).</em></p>

<br>

1. **Explore the images.** Checked 63,660 labelled images for class imbalance, size differences, missing files and duplicates.

2. **Prepare the data.** Resized and padded images without stretching them, normalised pixel values and added training augmentation. Used a stratified 80/20 split, keeping identical images together to reduce data leakage.

3. **Build a baseline.** Trained a custom CNN using convolution, ReLU, max pooling and dropout. Tracked training and validation loss to assess learning and overfitting.

4. **Apply transfer learning.** Replaced MobileNetV2's classifier and trained it with the pretrained features frozen. Then fine tuned selected feature layers using smaller learning rates.

5. **Evaluate and investigate.** Compared all three models on the same validation set using accuracy, F1 scores and confusion matrices. Inspected incorrect predictions, including confident mistakes and ambiguous crops.

6. **Check and submit.** Reloaded saved checkpoints to verify results, generated predictions for 29,443 test images and checked submission IDs, labels and coverage before submitting to Kaggle.

Read the [full report](report.pdf) for the experiments, error analysis and limitations.

<br>

## Results

<div align="center">
<table>
  <thead>
    <tr><th>Model</th><th>Validation accuracy</th><th>Kaggle score</th></tr>
  </thead>
  <tbody>
    <tr><td>Custom CNN</td><td align="center">83.78%</td><td align="center">Not submitted</td></tr>
    <tr><td>MobileNetV2, frozen features</td><td align="center">95.08%</td><td align="center">92.18%</td></tr>
    <tr><td>MobileNetV2, partial fine tuning</td><td align="center"><strong>98.63%</strong></td><td align="center"><strong>93.97%</strong></td></tr>
  </tbody>
</table>
</div>

<br>

## Running locally

Use Python 3.14. Create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
python -m jupyter lab
```

Download the [competition data](https://www.kaggle.com/competitions/enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch) into `data/raw/`, keeping its original folder structure.

Open the [notebooks](notebooks) in Jupyter. Use notebook **06** to evaluate the saved model without retraining, and enable `RUN_EVALUATION` in its configuration cell.
