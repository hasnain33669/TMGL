import argparse
import torch
import numpy as np
from tmgl import (
    TMGL, create_tmgl_model,
    TMGLMultiLabelTrainer,
    load_multilabel_data, create_dataloaders
)


def main():
    parser = argparse.ArgumentParser(description='Train TMGL for multi-label classification')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, default='data/', help='Data directory')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/best_multilabel_model.pt', help='Model save path')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    data_path = f"{args.data_path}/{args.dataset}.csv"
    df, target_columns = load_multilabel_data(data_path)
    train_loader, val_loader, test_loader = create_dataloaders(
        df, task='multilabel', batch_size=args.batch_size, target_columns=target_columns
    )
    
    # Compute class weights for imbalance
    pos_weight = None
    if args.use_class_weights:
        # Compute positive ratio for each label
        labels = np.array(df[target_columns].values)
        pos_ratios = labels.mean(axis=0)
        # pos_weight = (1 - pos_ratios) / pos_ratios
        pos_weight = torch.tensor((1 - pos_ratios) / pos_ratios, dtype=torch.float)
        print(f"Class weights computed: {pos_weight[:5]}...")
    
    # Create model
    model = create_tmgl_model(
        num_classes=len(target_columns),
        hidden_dim=args.hidden_dim,
        task='multilabel'
    )
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of labels: {len(target_columns)}")
    
    # Train
    trainer = TMGLMultiLabelTrainer(model, device, pos_weight=pos_weight)
    results = trainer.train(
        train_loader, val_loader, test_loader,
        epochs=args.epochs, save_path=args.save_path
    )
    
  


if __name__ == "__main__":
    main()
