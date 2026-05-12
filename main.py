from data_loader import load_data
from train import train_model
from evaluate import plot_curves, evaluate_model
from config import DEVICE

def main():
    print("Loading data...")
    train_loader, val_loader = load_data()

    print("Starting training...")
    model, history = train_model(train_loader, val_loader)

    print("Plotting curves...")
    plot_curves(history)

    print("Evaluating on validation set...")
    evaluate_model(model, val_loader, DEVICE)

if __name__ == "__main__":
    main()