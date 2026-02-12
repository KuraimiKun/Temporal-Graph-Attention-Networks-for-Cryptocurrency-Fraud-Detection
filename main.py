#!/usr/bin/env python3
"""
Ethereum Fraud Detection using Temporal Graph Networks

Main entry point for training, evaluation, and inference.

Usage:
    # Train a new model
    python main.py --mode train --config config/config.yaml
    
    # Evaluate a trained model
    python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
    
    # Generate explanations for fraud predictions
    python main.py --mode explain --checkpoint checkpoints/best_model.pt --node-id 12345
    
    # Run inference on new data
    python main.py --mode predict --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from utils.helpers import (
    load_config, 
    set_seed, 
    get_device, 
    setup_logging,
    create_experiment_dir,
    plot_training_history,
    count_parameters,
)
from data.elliptic_loader import (
    load_elliptic_dataset,
    create_temporal_splits,
)
from models.temporal_gnn import create_model
from losses.imbalanced_losses import create_loss_function
from training.trainer import TemporalGNNTrainer, TrainingConfig
from evaluation.metrics import FraudMetrics, create_metrics_summary_plot
from explainability.gnn_explainer import FraudExplainer, explain_prediction


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ethereum Fraud Detection with Temporal GNNs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train:      python main.py --mode train --config config/config.yaml
  Evaluate:   python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
  Explain:    python main.py --mode explain --node-id 12345
        """
    )
    
    # Mode
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['train', 'evaluate', 'explain', 'predict'],
        help='Operation mode'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    
    # Model checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint for evaluation/inference'
    )
    
    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/elliptic/',
        help='Directory containing the Elliptic dataset'
    )
    
    # Explainability
    parser.add_argument(
        '--node-id',
        type=int,
        default=None,
        help='Node ID to explain (for explain mode)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/',
        help='Directory for outputs (predictions, explanations)'
    )
    
    # Experiment tracking
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for this experiment'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for computation'
    )
    
    # Seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_experiment(args) -> Path:
    """Set up experiment directory and logging."""
    # Create experiment directory
    exp_dir = create_experiment_dir(
        base_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(
        log_file=str(exp_dir / "logs" / "experiment.log"),
        level=log_level,
    )
    
    # Set seeds
    set_seed(args.seed)
    
    return exp_dir


def get_device_from_args(args) -> torch.device:
    """Get compute device from arguments."""
    if args.device == 'auto':
        return get_device(prefer_gpu=True)
    else:
        return torch.device(args.device)


def train(args):
    """
    Train a Temporal GNN model on the Elliptic dataset.
    
    Pipeline:
    1. Load configuration
    2. Load and preprocess dataset
    3. Create temporal train/val/test splits
    4. Initialize model and trainer
    5. Train with early stopping
    6. Evaluate on test set
    7. Save final model and metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)
    
    # Setup
    exp_dir = setup_experiment(args)
    device = get_device_from_args(args)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load dataset
    logger.info("Loading Elliptic dataset...")
    try:
        dataset = load_elliptic_dataset(root=args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please download the Elliptic dataset first.")
        sys.exit(1)
    
    # Create temporal splits
    logger.info("Creating temporal train/val/test splits...")
    train_seq, val_seq, test_seq = create_temporal_splits(
        dataset,
        train_timesteps=config['data']['elliptic']['train_timesteps'],
        val_timesteps=config['data']['elliptic']['val_timesteps'],
        test_timesteps=config['data']['elliptic']['test_timesteps'],
    )
    
    # Log dataset statistics
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"  Train: {len(train_seq)} timesteps")
    logger.info(f"  Val: {len(val_seq)} timesteps")
    logger.info(f"  Test: {len(test_seq)} timesteps")
    logger.info(f"  Node features: {dataset.metadata['node_feature_dim']}")
    logger.info(f"  Fraud ratio: {dataset.fraud_ratio:.2%}")
    
    # Create model - use actual feature dimension from dataset
    logger.info("Creating model...")
    model_config = config['model']
    node_input_dim = dataset.metadata['node_feature_dim']  # Use actual dimension from data
    model = create_model(
        architecture=model_config['architecture'],
        node_input_dim=node_input_dim,
        edge_input_dim=model_config['edge_input_dim'],
        hidden_channels=model_config['gconv_gru']['hidden_channels'],
        num_classes=model_config['classifier']['num_classes'],
        dropout=model_config['gconv_gru']['dropout'],
    )
    
    params = count_parameters(model)
    logger.info(f"Model created: {model_config['architecture']}")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Trainable parameters: {params['trainable']:,}")
    
    # Create training configuration
    train_config = TrainingConfig(
        device=str(device),
        seed=args.seed,
        learning_rate=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        epochs=config['training']['epochs'],
        loss_type=config['loss']['type'],
        focal_alpha=config['loss']['focal_loss']['alpha'],
        focal_gamma=config['loss']['focal_loss']['gamma'],
        early_stopping=config['training']['early_stopping']['enabled'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        checkpoint_dir=str(exp_dir / "checkpoints"),
        gradient_clip=config['training']['gradient_clip'],
    )
    
    # Create trainer
    class_counts = [dataset.metadata['num_legit'], dataset.metadata['num_fraud']]
    trainer = TemporalGNNTrainer(model, train_config, class_counts=class_counts)
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(train_seq, val_seq)
    
    # Plot training history
    plot_training_history(
        history,
        save_path=str(exp_dir / "plots" / "training_history.png")
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_seq)
    
    # Compute full metrics
    probs, preds = trainer.predict(test_seq)
    final_snapshot = test_seq[-1]
    mask = final_snapshot.train_mask.numpy()
    y_true = final_snapshot.y.numpy()[mask]
    probs_masked = probs[mask]
    
    full_metrics = FraudMetrics.compute(y_true, probs_masked)
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(str(full_metrics))
    
    # Save metrics summary plot
    create_metrics_summary_plot(
        full_metrics,
        save_path=str(exp_dir / "plots" / "test_metrics.png")
    )
    
    # Save final results
    results = {
        'test_metrics': full_metrics.to_dict(),
        'training_history': history,
        'config': config,
    }
    torch.save(results, exp_dir / "results.pt")
    
    logger.info(f"\nExperiment completed! Results saved to: {exp_dir}")
    logger.info(f"Best model checkpoint: {exp_dir / 'checkpoints' / 'best_model.pt'}")
    
    return full_metrics


def evaluate(args):
    """
    Evaluate a trained model on the test set.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)
    
    if args.checkpoint is None:
        logger.error("Please provide a checkpoint path with --checkpoint")
        sys.exit(1)
    
    # Setup
    device = get_device_from_args(args)
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load dataset
    logger.info("Loading Elliptic dataset...")
    dataset = load_elliptic_dataset(root=args.data_dir)
    
    # Get test set
    _, _, test_seq = create_temporal_splits(
        dataset,
        train_timesteps=config['data']['elliptic']['train_timesteps'],
        val_timesteps=config['data']['elliptic']['val_timesteps'],
        test_timesteps=config['data']['elliptic']['test_timesteps'],
    )
    
    # Create model
    model_config = config['model']
    model = create_model(
        architecture=model_config['architecture'],
        node_input_dim=model_config['node_input_dim'],
        hidden_channels=model_config['gconv_gru']['hidden_channels'],
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded checkpoint from: {args.checkpoint}")
    logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    
    # Create trainer for evaluation
    train_config = TrainingConfig(device=str(device))
    trainer = TemporalGNNTrainer(model, train_config)
    
    # Evaluate
    test_metrics = trainer.evaluate(test_seq)
    
    # Full metrics
    probs, preds = trainer.predict(test_seq)
    final_snapshot = test_seq[-1]
    mask = final_snapshot.train_mask.numpy()
    y_true = final_snapshot.y.numpy()[mask]
    probs_masked = probs[mask]
    
    full_metrics = FraudMetrics.compute(y_true, probs_masked)
    
    logger.info("\n" + str(full_metrics))
    
    return full_metrics


def explain(args):
    """
    Generate explanations for fraud predictions.
    """
    logger.info("=" * 60)
    logger.info("EXPLANATION MODE")
    logger.info("=" * 60)
    
    if args.node_id is None:
        logger.error("Please provide a node ID to explain with --node-id")
        sys.exit(1)
    
    # Setup
    device = get_device_from_args(args)
    output_dir = Path(args.output_dir) / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_elliptic_dataset(root=args.data_dir)
    _, _, test_seq = create_temporal_splits(dataset)
    
    # Create and load model with actual feature dimension from data
    model_config = config['model']
    node_input_dim = dataset.metadata['node_feature_dim']  # Use actual dimension
    model = create_model(
        architecture=model_config['architecture'],
        node_input_dim=node_input_dim,
        edge_input_dim=model_config['edge_input_dim'],
        hidden_channels=model_config['gconv_gru']['hidden_channels'],
        num_classes=model_config['classifier']['num_classes'],
        dropout=model_config['gconv_gru']['dropout'],
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Get final snapshot for explanation
    final_snapshot = test_seq[-1]
    x = final_snapshot.x.to(device)
    edge_index = final_snapshot.edge_index.to(device)
    
    # Generate explanation
    logger.info(f"Generating explanation for node {args.node_id}...")
    
    explainer = FraudExplainer(model, epochs=200, lr=0.01)
    explanation = explainer.explain_fraud_prediction(
        node_idx=args.node_id,
        x=x,
        edge_index=edge_index,
        top_k_edges=10,
        top_k_features=5,
    )
    
    # Visualize
    save_path = output_dir / f"fraud_explanation_node_{args.node_id}.png"
    explainer.visualize(
        explanation=explanation,
        x=x,
        edge_index=edge_index,
        node_labels=final_snapshot.y.to(device),
        save_path=str(save_path),
    )
    
    logger.info(f"Explanation saved to: {save_path}")
    
    # Print summary
    logger.info("\nExplanation Summary:")
    logger.info(f"  Target node: {explanation['node_idx']}")
    logger.info(f"  Prediction: {explanation['prediction']}")
    logger.info(f"  Subgraph nodes: {len(explanation['subgraph_nodes'])}")
    
    if explanation['top_edges'] is not None:
        logger.info(f"  Top important edges: {explanation['top_edges'].shape[1]}")
    
    return explanation


def predict(args):
    """
    Run inference on new data.
    """
    logger.info("=" * 60)
    logger.info("PREDICTION MODE")
    logger.info("=" * 60)
    
    if args.checkpoint is None:
        logger.error("Please provide a checkpoint path with --checkpoint")
        sys.exit(1)
    
    # This is a placeholder for inference on new data
    # In practice, you would:
    # 1. Load data from BigQuery or Alchemy API
    # 2. Preprocess into temporal graph format
    # 3. Run inference with the trained model
    # 4. Save predictions
    
    logger.info("Prediction mode requires live data integration.")
    logger.info("See data/bigquery_loader.py for BigQuery integration.")
    logger.info("See data/alchemy_loader.py for Alchemy API integration.")
    
    # Example pseudocode:
    # from data.bigquery_loader import BigQueryEthereumLoader
    # loader = BigQueryEthereumLoader(project_id="my-project")
    # df = loader.fetch_transactions(start_block, end_block)
    # graph_data = loader.build_graph_from_transactions(df)
    # predictions = model(graph_data['x'], graph_data['edge_index'])


def main():
    """Main entry point."""
    args = parse_args()
    
    # Route to appropriate handler
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'explain':
        explain(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
