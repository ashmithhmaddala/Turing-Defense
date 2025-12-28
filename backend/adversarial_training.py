"""
Adversarial Training System
============================
Makes the bot detection model robust against evasion attempts.

Techniques:
1. Gradient-based adversarial perturbations (FGSM, PGD)
2. Feature-level noise injection
3. Mixup and CutMix augmentation
4. Curriculum learning (easy → hard samples)
5. GAN-based adversarial generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import random

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


class AdversarialTrainer:
    """
    Adversarial training wrapper for bot detection models.
    Makes models robust against sophisticated evasion attempts.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        attack_steps: int = 10,
        attack_lr: float = 0.01
    ):
        """
        Args:
            model: PyTorch neural network model
            epsilon: Maximum perturbation size (L-infinity norm)
            attack_steps: Number of PGD attack steps
            attack_lr: Learning rate for attack optimization
        """
        self.model = model
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.attack_lr = attack_lr
        
    def fgsm_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: Optional[float] = None
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method attack.
        Creates adversarial examples by perturbing inputs in the direction
        of the gradient of the loss.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = F.binary_cross_entropy(outputs.squeeze(), y.float())
        
        # Backward pass
        loss.backward()
        
        # Create adversarial perturbation
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x + perturbation
        
        # Clamp to valid range
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()
    
    def pgd_attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: Optional[float] = None,
        steps: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Projected Gradient Descent attack.
        More powerful iterative attack that creates stronger adversarial examples.
        """
        if epsilon is None:
            epsilon = self.epsilon
        if steps is None:
            steps = self.attack_steps
        if alpha is None:
            alpha = epsilon / steps * 2
        
        # Start from random point within epsilon ball
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            loss = F.binary_cross_entropy(outputs.squeeze(), y.float())
            
            # Backward pass
            loss.backward()
            
            # Update adversarial example
            x_adv = x_adv + alpha * x_adv.grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv.detach()
    
    def train_adversarial(
        self,
        train_loader: 'torch.utils.data.DataLoader',
        optimizer: torch.optim.Optimizer,
        epochs: int = 50,
        attack_ratio: float = 0.5,
        curriculum: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train model with adversarial examples.
        
        Args:
            train_loader: PyTorch DataLoader with training data
            optimizer: PyTorch optimizer
            epochs: Number of training epochs
            attack_ratio: Fraction of batch to attack (0.5 = 50% adversarial)
            curriculum: If True, start with weak attacks and increase strength
            verbose: Print training progress
            
        Returns:
            Training history with loss and accuracy
        """
        history = {
            "loss": [],
            "accuracy": [],
            "adv_loss": [],
            "adv_accuracy": []
        }
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            epoch_adv_loss = 0
            epoch_adv_correct = 0
            
            # Curriculum: increase attack strength over time
            if curriculum:
                current_epsilon = self.epsilon * min(1.0, (epoch + 1) / (epochs * 0.5))
            else:
                current_epsilon = self.epsilon
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Split batch: clean + adversarial
                split_idx = int(len(batch_x) * (1 - attack_ratio))
                
                # Clean examples
                clean_x = batch_x[:split_idx]
                clean_y = batch_y[:split_idx]
                
                # Adversarial examples
                adv_x = batch_x[split_idx:]
                adv_y = batch_y[split_idx:]
                
                # Generate adversarial examples
                self.model.eval()
                adv_x = self.pgd_attack(adv_x, adv_y, epsilon=current_epsilon)
                self.model.train()
                
                # Forward pass on combined batch
                combined_x = torch.cat([clean_x, adv_x])
                combined_y = torch.cat([clean_y, adv_y])
                
                outputs = self.model(combined_x)
                loss = F.binary_cross_entropy(outputs.squeeze(), combined_y.float())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                predictions = (outputs.squeeze() > 0.5).long()
                epoch_loss += loss.item() * len(combined_x)
                epoch_correct += (predictions == combined_y).sum().item()
                epoch_total += len(combined_x)
                
                # Track adversarial accuracy separately
                adv_outputs = self.model(adv_x)
                adv_predictions = (adv_outputs.squeeze() > 0.5).long()
                epoch_adv_correct += (adv_predictions == adv_y).sum().item()
                epoch_adv_loss += F.binary_cross_entropy(
                    adv_outputs.squeeze(), adv_y.float()
                ).item() * len(adv_x)
            
            # Epoch metrics
            epoch_loss /= epoch_total
            epoch_accuracy = epoch_correct / epoch_total
            epoch_adv_loss /= (epoch_total * attack_ratio)
            epoch_adv_accuracy = epoch_adv_correct / (epoch_total * attack_ratio)
            
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_accuracy)
            history["adv_loss"].append(epoch_adv_loss)
            history["adv_accuracy"].append(epoch_adv_accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"[ADV-TRAIN] Epoch {epoch+1}/{epochs}: "
                      f"Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.3f}, "
                      f"Adv Loss: {epoch_adv_loss:.4f}, Adv Acc: {epoch_adv_accuracy:.3f}, "
                      f"Epsilon: {current_epsilon:.4f}")
        
        return history


class FeatureAugmentor:
    """
    Data augmentation for behavioral features.
    Creates variations of training data to improve robustness.
    """
    
    @staticmethod
    def add_noise(
        features: np.ndarray,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    @staticmethod
    def temporal_jitter(
        features: np.ndarray,
        jitter_factor: float = 0.1
    ) -> np.ndarray:
        """Add timing jitter to temporal features"""
        jitter = np.random.uniform(1 - jitter_factor, 1 + jitter_factor, features.shape)
        return features * jitter
    
    @staticmethod
    def feature_dropout(
        features: np.ndarray,
        dropout_rate: float = 0.1
    ) -> np.ndarray:
        """Randomly zero out features"""
        mask = np.random.binomial(1, 1 - dropout_rate, features.shape)
        return features * mask
    
    @staticmethod
    def mixup(
        features1: np.ndarray,
        features2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: blend two samples together.
        Creates smoother decision boundaries.
        """
        lam = np.random.beta(alpha, alpha)
        mixed_features = lam * features1 + (1 - lam) * features2
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        return mixed_features, mixed_labels
    
    @staticmethod
    def cutmix(
        features1: np.ndarray,
        features2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix augmentation: replace feature subset from another sample.
        """
        lam = np.random.beta(alpha, alpha)
        
        # Random feature indices to replace
        num_features = features1.shape[-1]
        num_cut = int(num_features * (1 - lam))
        cut_indices = np.random.choice(num_features, num_cut, replace=False)
        
        # Create mixed sample
        mixed_features = features1.copy()
        mixed_features[..., cut_indices] = features2[..., cut_indices]
        
        # Adjust labels proportionally
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_features, mixed_labels
    
    def augment_batch(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentations to a batch.
        
        Args:
            features: (batch_size, num_features) feature array
            labels: (batch_size,) label array
            augmentation_factor: How many augmented copies to create
            
        Returns:
            Augmented features and labels
        """
        all_features = [features]
        all_labels = [labels]
        
        for _ in range(augmentation_factor - 1):
            aug_features = features.copy()
            
            # Apply random augmentations
            if random.random() < 0.5:
                aug_features = self.add_noise(aug_features, noise_level=0.05)
            
            if random.random() < 0.5:
                aug_features = self.temporal_jitter(aug_features, jitter_factor=0.1)
            
            if random.random() < 0.3:
                aug_features = self.feature_dropout(aug_features, dropout_rate=0.1)
            
            # Mixup with random partner
            if random.random() < 0.3:
                partner_idx = np.random.permutation(len(features))
                aug_features, aug_labels = self.mixup(
                    aug_features, features[partner_idx],
                    labels, labels[partner_idx]
                )
                all_labels.append(aug_labels)
            else:
                all_labels.append(labels.copy())
            
            all_features.append(aug_features)
        
        return np.vstack(all_features), np.hstack(all_labels)


class CurriculumLearning:
    """
    Curriculum learning: train on easy samples first, then harder ones.
    This helps the model learn robust features progressively.
    """
    
    def __init__(
        self,
        model: nn.Module,
        difficulty_scorer: Optional[Callable] = None
    ):
        """
        Args:
            model: PyTorch model
            difficulty_scorer: Function that scores sample difficulty (0-1)
        """
        self.model = model
        self.difficulty_scorer = difficulty_scorer or self._default_scorer
        
    def _default_scorer(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Score difficulty based on prediction confidence.
        Lower confidence = harder sample.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features)
            outputs = self.model(x).squeeze().numpy()
        
        # Difficulty = distance from correct prediction
        difficulties = np.abs(outputs - labels)
        return difficulties
    
    def get_curriculum_batches(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_stages: int = 5,
        batch_size: int = 32
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create curriculum batches from easy to hard.
        
        Args:
            features: Training features
            labels: Training labels
            num_stages: Number of curriculum stages
            batch_size: Batch size for each stage
            
        Returns:
            List of (features, labels) tuples, ordered easy → hard
        """
        # Score all samples
        difficulties = self.difficulty_scorer(features, labels)
        
        # Sort by difficulty
        sorted_indices = np.argsort(difficulties)
        
        # Split into stages
        stage_size = len(sorted_indices) // num_stages
        batches = []
        
        for stage in range(num_stages):
            start_idx = stage * stage_size
            end_idx = (stage + 1) * stage_size if stage < num_stages - 1 else len(sorted_indices)
            
            stage_indices = sorted_indices[start_idx:end_idx]
            stage_features = features[stage_indices]
            stage_labels = labels[stage_indices]
            
            # Create mini-batches within stage
            for i in range(0, len(stage_indices), batch_size):
                batch_features = stage_features[i:i+batch_size]
                batch_labels = stage_labels[i:i+batch_size]
                batches.append((batch_features, batch_labels))
        
        return batches


class RobustTrainingPipeline:
    """
    Complete robust training pipeline combining all techniques:
    - Adversarial training
    - Data augmentation
    - Curriculum learning
    - Ensemble methods
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Callable,
        device: str = "cpu"
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        
        self.adversarial_trainer = None
        self.feature_augmentor = FeatureAugmentor()
        self.curriculum = None
        
    def train(
        self,
        train_sessions: List[Dict],
        train_labels: List[int],
        val_sessions: Optional[List[Dict]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        adversarial_ratio: float = 0.3,
        augmentation_factor: int = 2,
        use_curriculum: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Full robust training pipeline.
        
        Args:
            train_sessions: Training session data
            train_labels: Training labels (0=human, 1=bot)
            val_sessions: Validation sessions (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            adversarial_ratio: Fraction of adversarial examples (0-1)
            augmentation_factor: Data augmentation multiplier
            use_curriculum: Whether to use curriculum learning
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")
        
        print("=" * 60)
        print("ROBUST ADVERSARIAL TRAINING PIPELINE")
        print("=" * 60)
        
        # Extract features
        print("\n[1/4] Extracting features...")
        X_train = np.array([self.feature_extractor(s) for s in train_sessions])
        y_train = np.array(train_labels)
        
        if val_sessions:
            X_val = np.array([self.feature_extractor(s) for s in val_sessions])
            y_val = np.array(val_labels)
        
        # Data augmentation
        print(f"[2/4] Augmenting data (factor: {augmentation_factor}x)...")
        X_train_aug, y_train_aug = self.feature_augmentor.augment_batch(
            X_train, y_train, augmentation_factor
        )
        print(f"      Original: {len(X_train)} samples → Augmented: {len(X_train_aug)} samples")
        
        # Normalize features
        self.feature_mean = X_train_aug.mean(axis=0)
        self.feature_std = X_train_aug.std(axis=0) + 1e-8
        X_train_norm = (X_train_aug - self.feature_mean) / self.feature_std
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.LongTensor(y_train_aug.astype(int)).to(self.device)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Setup adversarial trainer
        print(f"[3/4] Setting up adversarial trainer (ratio: {adversarial_ratio})...")
        self.adversarial_trainer = AdversarialTrainer(
            self.model,
            epsilon=0.1,
            attack_steps=10,
            attack_lr=0.01
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training
        print(f"[4/4] Training for {epochs} epochs...")
        print("-" * 60)
        
        self.model.to(self.device)
        history = self.adversarial_trainer.train_adversarial(
            train_loader,
            optimizer,
            epochs=epochs,
            attack_ratio=adversarial_ratio,
            curriculum=use_curriculum,
            verbose=verbose
        )
        
        # Validation
        if val_sessions:
            print("\n[VALIDATION]")
            X_val_norm = (X_val - self.feature_mean) / self.feature_std
            X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor).squeeze()
                val_preds = (val_outputs > 0.5).cpu().numpy()
                val_acc = (val_preds == y_val).mean()
                print(f"Validation Accuracy: {val_acc:.1%}")
                history["val_accuracy"] = val_acc
        
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        return history


def test_adversarial_robustness(
    model: nn.Module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3]
) -> Dict[float, float]:
    """
    Test model robustness against adversarial attacks at different strengths.
    
    Returns:
        Dictionary mapping epsilon → accuracy
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    
    results = {}
    trainer = AdversarialTrainer(model, epsilon=0.1)
    
    X_tensor = torch.FloatTensor(test_features)
    y_tensor = torch.LongTensor(test_labels)
    
    # Clean accuracy
    model.eval()
    with torch.no_grad():
        clean_outputs = model(X_tensor)
        clean_preds = (clean_outputs.squeeze() > 0.5).numpy()
        clean_acc = (clean_preds == test_labels).mean()
    
    results[0.0] = clean_acc
    print(f"Clean Accuracy: {clean_acc:.1%}")
    
    # Adversarial accuracies
    for epsilon in epsilon_values:
        adv_X = trainer.pgd_attack(X_tensor, y_tensor, epsilon=epsilon)
        
        with torch.no_grad():
            adv_outputs = model(adv_X)
            adv_preds = (adv_outputs.squeeze() > 0.5).numpy()
            adv_acc = (adv_preds == test_labels).mean()
        
        results[epsilon] = adv_acc
        print(f"Adversarial Accuracy (ε={epsilon}): {adv_acc:.1%}")
    
    return results


if __name__ == "__main__":
    print("Adversarial Training Module")
    print("=" * 60)
    print("Available components:")
    print("  - AdversarialTrainer: FGSM/PGD attacks for robust training")
    print("  - FeatureAugmentor: Data augmentation techniques")
    print("  - CurriculumLearning: Easy-to-hard training schedule")
    print("  - RobustTrainingPipeline: Complete training system")
    print("  - test_adversarial_robustness: Evaluate attack resistance")
