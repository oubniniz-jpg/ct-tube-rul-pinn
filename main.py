"""
CT-Scanner X-Ray Tube Predictive Maintenance
Physics-Informed Deep Learning for RUL Prediction

Main entry point - run the full pipeline:
1. Generate synthetic data via physics simulation
2. Preprocess and create sequences
3. Train LSTM model with physics-informed loss
4. Evaluate and visualize results

Author: OUBNINI Zakaria
Supervisor: Mr. Feng Wang
Henan University of Technology - 2026
"""

import yaml
import torch
import numpy as np
from src.simulation import DegradationEngine
from src.preprocessing import DataLoader, FeatureEngineer
from src.models import LSTMPredictor, CNNLSTMPredictor
from src.utils import Trainer, compute_metrics, plot_degradation, plot_rul_predictions


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("CT-Scanner X-Ray Tube - Predictive Maintenance System")
    print("Physics-Informed Deep Learning for RUL Prediction")
    print("=" * 60)

    # Load configuration
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic degradation data...")
    engine = DegradationEngine(config)
    dataset = engine.generate_dataset(
        num_tubes=config["simulation"]["num_tubes"],
        save_path="data/synthetic/degradation_data.csv"
    )
    print(f"  Dataset shape: {dataset.shape}")
    print(f"  Tubes simulated: {dataset['tube_id'].nunique()}")

    # Plot example degradation
    plot_degradation(dataset, tube_id=0, save_path="results/figures/degradation_example.png")

    # Step 2: Feature engineering
    print("\n[Step 2] Engineering features...")
    fe = FeatureEngineer()
    dataset = fe.add_features(dataset)

    # Step 3: Prepare data
    print("\n[Step 3] Preparing dataset...")
    loader = DataLoader(config)
    dataloaders = loader.prepare_dataset(dataset)
    for split, dl in dataloaders.items():
        print(f"  {split}: {len(dl.dataset)} samples")

    # Step 4: Build model
    print("\n[Step 4] Building model...")
    model_type = config["model"]["type"]
    if model_type == "cnn_lstm":
        model = CNNLSTMPredictor(config)
    else:
        model = LSTMPredictor(config)
    print(f"  Model: {model_type.upper()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Step 5: Train
    print("\n[Step 5] Training model...")
    trainer = Trainer(model, config, device=device)
    history = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        save_path="results/best_model.pt"
    )

    # Step 6: Evaluate
    print("\n[Step 6] Evaluating on test set...")
    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloaders["test"]:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(y_batch.numpy().flatten())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    print(f"\n  Results:")
    for name, value in metrics.items():
        print(f"    {name.upper()}: {value:.4f}")

    # Plot results
    plot_rul_predictions(y_true, y_pred, save_path="results/figures/rul_predictions.png")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
