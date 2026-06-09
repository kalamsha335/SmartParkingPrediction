import os
import json
import matplotlib.pyplot as plt
import argparse

def plot_training_history(history_path, save_dir):
    """
    Read the JSON training history and generate graphs for loss and accuracy.
    """
    if not os.path.exists(history_path):
        print(f"Error: History file not found at {history_path}")
        print("Please train the model first to generate training history.")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot Accuracy vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy vs Epoch', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Save accuracy plot
    acc_plot_path = os.path.join(save_dir, 'accuracy_vs_epoch.png')
    plt.savefig(acc_plot_path)
    print(f"Accuracy plot saved to {acc_plot_path}")
    plt.show()

    # Plot Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss vs Epoch', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Save loss plot
    loss_plot_path = os.path.join(save_dir, 'loss_vs_epoch.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_history = os.path.join(script_dir, '../model/training_history.json')
    
    parser = argparse.ArgumentParser(description="Visualize CNN Training Results")
    parser.add_argument('--history_path', type=str, default=default_history, help='Path to the history JSON file')
    
    args = parser.parse_args()
    
    # Use the visualization directory to save plots by default
    save_dir = '.'
    plot_training_history(args.history_path, save_dir)

if __name__ == "__main__":
    main()
