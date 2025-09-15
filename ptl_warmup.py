import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Optional, List
import math

class NonLinearWarmupScheduler(LambdaLR):
    """
    A learning rate scheduler with non-linear warmup compatible with PyTorch Lightning.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        warmup_func: Callable[[torch.Tensor], torch.Tensor],
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.warmup_func = warmup_func
        super().__init__(optimizer, self.lr_lambda, last_epoch)
    
    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            # Normalize step to [0, 1] range
            normalized_step = torch.tensor(float(step) / self.warmup_steps)
            # Apply warmup function and normalize to [0, 1]
            warmup_value = self.warmup_func(normalized_step)
            # Ensure output is in [0, 1] range
            return float(warmup_value)
        return 1.0

class NonLinearWarmupWrapper:
    """
    Wrapper for easy integration with PyTorch Lightning.
    """
    
    def __init__(
        self,
        warmup_steps: int,
        warmup_func: Callable[[torch.Tensor], torch.Tensor],
        scheduler: Optional[Callable] = None
    ):
        self.warmup_steps = warmup_steps
        self.warmup_func = warmup_func
        self.scheduler = scheduler
        self._scheduler_instance = None
    
    def setup(self, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        # Create warmup scheduler
        warmup_scheduler = NonLinearWarmupScheduler(
            optimizer,
            self.warmup_steps,
            self.warmup_func
        )
        
        # If main scheduler is provided, chain them
        if self.scheduler:
            self._scheduler_instance = self.scheduler(optimizer, **kwargs)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, self._scheduler_instance],
                milestones=[self.warmup_steps]
            )
        else:
            return warmup_scheduler

class ExampleModel(pl.LightningModule):
    def __init__(self, warmup_type="sqrt"):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.criterion = torch.nn.MSELoss()
        
        # Store warmup type for identification
        self.warmup_type = warmup_type
        
        # Define different warmup configurations
        if warmup_type == "sqrt":
            self.warmup_scheduler = NonLinearWarmupWrapper(
                warmup_steps=1000,
                warmup_func=torch.sqrt
            )
        elif warmup_type == "square":
            self.warmup_scheduler = NonLinearWarmupWrapper(
                warmup_steps=1000,
                warmup_func=torch.square
            )
        elif warmup_type == "linear":
            self.warmup_scheduler = NonLinearWarmupWrapper(
                warmup_steps=1000,
                warmup_func=lambda x: x  # Linear
            )
        else:  # custom sigmoid
            def sigmoid_warmup(x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(6 * (x - 0.5))
            self.warmup_scheduler = NonLinearWarmupWrapper(
                warmup_steps=1000,
                warmup_func=sigmoid_warmup
            )
    
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = self.warmup_scheduler.setup(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# Create dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000):
        self.size = size
        # Generate random data
        torch.manual_seed(42)
        self.x = torch.randn(size, 10)
        self.y = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Custom callback to collect learning rate data
class LRCallback(pl.Callback):
    def __init__(self):
        self.lrs = []
        self.steps = []
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Record learning rate every step
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        self.lrs.append(current_lr)
        self.steps.append(trainer.global_step)

def train_single_model(warmup_type="sqrt", max_steps=2000):
    """Train a single model and return LR history"""
    
    # Create dataset
    train_dataset = DummyDataset(5000)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = ExampleModel(warmup_type=warmup_type)
    
    # Create LR callback
    lr_callback = LRCallback()
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        log_every_n_steps=1,
        accelerator="auto",
        devices=1,
        logger=False,  # Disable default logger
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[lr_callback]
    )
    
    # Train model
    trainer.fit(model, train_loader)
    
    return lr_callback.steps, lr_callback.lrs

def plot_theoretical_curves():
    """Plot theoretical warmup curves without training"""
    warmup_steps = 1000
    steps = np.arange(0, 1500)  # Extend beyond warmup for visualization
    
    # Define warmup functions
    def linear_warmup(step):
        return min(1.0, step / warmup_steps)
    
    def sqrt_warmup(step):
        if step < warmup_steps:
            normalized = step / warmup_steps
            return np.sqrt(normalized)
        return 1.0
    
    def square_warmup(step):
        if step < warmup_steps:
            normalized = step / warmup_steps
            return normalized ** 2
        return 1.0
    
    def sigmoid_warmup(step):
        if step < warmup_steps:
            normalized = step / warmup_steps
            return 1 / (1 + np.exp(-6 * (normalized - 0.5)))
        return 1.0
    
    # Calculate LR values
    linear_lrs = [linear_warmup(step) for step in steps]
    sqrt_lrs = [sqrt_warmup(step) for step in steps]
    square_lrs = [square_warmup(step) for step in steps]
    sigmoid_lrs = [sigmoid_warmup(step) for step in steps]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(steps, linear_lrs, label="Linear", linewidth=2)
    plt.plot(steps, sqrt_lrs, label="Square Root", linewidth=2)
    plt.plot(steps, square_lrs, label="Quadratic", linewidth=2)
    plt.plot(steps, sigmoid_lrs, label="Sigmoid", linewidth=2)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate Scale (0-1)")
    plt.title("Theoretical Non-Linear Warmup Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.7, label=f"Warmup End ({warmup_steps} steps)")
    plt.tight_layout()
    
    # Save plot
    plt.savefig("theoretical_warmup_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def compare_warmup_strategies():
    """Compare different warmup strategies by training models"""
    
    # Different warmup types to compare
    warmup_types = ["linear", "sqrt", "square", "sigmoid"]
    results = {}
    
    print("Training models with different warmup strategies...")
    
    # Train models with different warmup strategies
    for i, warmup_type in enumerate(warmup_types):
        print(f"{i+1}/{len(warmup_types)}: Training with {warmup_type} warmup...")
        steps, lrs = train_single_model(warmup_type, max_steps=1500)
        results[warmup_type] = {"steps": steps, "lrs": lrs}
        print(f"  Collected {len(steps)} data points")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red', 'orange']
    for i, (warmup_type, data) in enumerate(results.items()):
        plt.plot(data["steps"], data["lrs"], 
                label=f"{warmup_type.capitalize()} warmup", 
                linewidth=2, 
                color=colors[i])
    
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Actual Learning Rate Schedules from Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1000, color='red', linestyle='--', alpha=0.7, label="Warmup End (1000 steps)")
    plt.tight_layout()
    
    # Save plot
    plt.savefig("lr_warmup_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final statistics
    print("\nFinal Learning Rates:")
    for warmup_type, data in results.items():
        if data["lrs"]:
            print(f"{warmup_type.capitalize()} warmup: {data['lrs'][-1]:.6f}")

def plot_detailed_comparison():
    """Create a more detailed comparison with theoretical curves"""
    
    # Plot theoretical curves
    warmup_steps = 1000
    steps = np.arange(0, 1500)
    
    # Define theoretical functions
    def linear_warmup(step):
        return min(1.0, step / warmup_steps)
    
    def sqrt_warmup(step):
        if step < warmup_steps:
            return np.sqrt(step / warmup_steps)
        return 1.0
    
    def square_warmup(step):
        if step < warmup_steps:
            normalized = step / warmup_steps
            return normalized ** 2
        return 1.0
    
    def sigmoid_warmup(step):
        if step < warmup_steps:
            normalized = step / warmup_steps
            return 1 / (1 + np.exp(-6 * (normalized - 0.5)))
        return 1.0
    
    # Calculate theoretical values
    theoretical_data = {
        "linear": [linear_warmup(step) for step in steps],
        "sqrt": [sqrt_warmup(step) for step in steps],
        "square": [square_warmup(step) for step in steps],
        "sigmoid": [sigmoid_warmup(step) for step in steps]
    }
    
    # Train actual models and get real data
    print("Collecting actual training data...")
    actual_data = {}
    for warmup_type in ["linear", "sqrt", "square", "sigmoid"]:
        print(f"Training {warmup_type} model...")
        steps_actual, lrs_actual = train_single_model(warmup_type, max_steps=1500)
        actual_data[warmup_type] = {"steps": steps_actual, "lrs": lrs_actual}
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = ['blue', 'green', 'red', 'orange']
    warmup_names = ["Linear", "Square Root", "Quadratic", "Sigmoid"]
    
    for i, warmup_type in enumerate(["linear", "sqrt", "square", "sigmoid"]):
        ax = axes[i]
        
        # Plot theoretical curve
        ax.plot(steps, theoretical_data[warmup_type], 
                '--', color=colors[i], linewidth=2, alpha=0.8,
                label=f"Theoretical {warmup_names[i]}")
        
        # Plot actual training data
        if warmup_type in actual_data:
            ax.plot(actual_data[warmup_type]["steps"], actual_data[warmup_type]["lrs"],
                    '-', color=colors[i], linewidth=2,
                    label=f"Actual {warmup_names[i]}")
        
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Learning Rate Scale")
        ax.set_title(f"{warmup_names[i]} Warmup")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1000, color='black', linestyle=':', alpha=0.7, label="Warmup End")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("detailed_warmup_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("1. Plotting theoretical warmup curves...")
    plot_theoretical_curves()
    
    print("\n2. Comparing actual warmup strategies...")
    compare_warmup_strategies()
    
    print("\n3. Creating detailed comparison...")
    plot_detailed_comparison()
    
    print("\nAll visualizations have been saved!")
