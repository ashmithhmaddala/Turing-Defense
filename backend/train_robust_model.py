"""
Complete Robust Training Pipeline
==================================
Trains bot detection model using:
1. Real datasets (CMU Keystroke + Balabit Mouse)
2. Attack simulations (9 different bot types)
3. Adversarial training (FGSM/PGD attacks)
4. Data augmentation (mixup, noise, dropout)
5. Curriculum learning (easy → hard)

This creates a production-ready model that's resistant to sophisticated evasion.
"""

import os
import sys
import numpy as np
import random

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loaders import load_and_prepare_training_data
from attack_simulators import AttackSimulator, BOT_PROFILES, generate_attack_dataset
from ml_neural_network import NeuralBotDetector, AdvancedFeatureExtractor, TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

import torch


def main():
    print("=" * 70)
    print("   ROBUST ADVERSARIAL TRAINING PIPELINE")
    print("   Turing Defense - Bot Detection System")
    print("=" * 70)
    
    # =========================================================
    # PHASE 1: Load Real Human Data
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LOADING REAL HUMAN DATA")
    print("=" * 70)
    
    # Load the real datasets (includes both human and synthetic bot samples)
    all_sessions, all_labels_raw = load_and_prepare_training_data(
        balabit_path="datasets/balabit",
        cmu_path="datasets/DSL-StrongPasswordData.csv",
        max_samples_per_source=300  # Use subset for speed
    )
    
    # Extract just the human sessions (label = 0)
    human_sessions = [s for s, l in zip(all_sessions, all_labels_raw) if l == 0]
    human_labels = [0] * len(human_sessions)
    print(f"\n[OK] Loaded {len(human_sessions)} real human sessions")
    
    # =========================================================
    # PHASE 2: Generate Attack Simulations
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: GENERATING ATTACK SIMULATIONS")
    print("=" * 70)
    
    # Create attack simulator with human templates for adversarial samples
    simulator = AttackSimulator(human_templates=human_sessions[:50])
    
    # Generate samples from each bot profile
    bot_sessions = []
    bot_labels = []
    
    samples_per_type = max(10, len(human_sessions) // len(BOT_PROFILES))
    
    for profile_name in BOT_PROFILES.keys():
        print(f"  Generating {samples_per_type} samples: {profile_name}")
        
        for _ in range(samples_per_type):
            session = simulator.generate_session(
                profile_name,
                duration_ms=random.randint(10000, 45000),
                num_clicks=random.randint(5, 20),
                num_keystrokes=random.randint(20, 80)
            )
            bot_sessions.append(session)
            bot_labels.append(1)
    
    # Generate adversarial samples (hardest to detect)
    print(f"\n  Generating {samples_per_type * 2} adversarial samples...")
    adv_sessions, adv_labels = simulator.generate_adversarial_samples(
        human_sessions[:100],
        num_samples=samples_per_type * 2
    )
    bot_sessions.extend(adv_sessions)
    bot_labels.extend(adv_labels)
    
    print(f"\n[OK] Generated {len(bot_sessions)} bot samples across {len(BOT_PROFILES) + 1} attack types")
    
    # =========================================================
    # PHASE 3: Combine and Balance Dataset
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: PREPARING BALANCED DATASET")
    print("=" * 70)
    
    # Balance classes
    min_class_size = min(len(human_sessions), len(bot_sessions))
    
    # Shuffle and sample
    human_indices = np.random.permutation(len(human_sessions))[:min_class_size]
    bot_indices = np.random.permutation(len(bot_sessions))[:min_class_size]
    
    balanced_sessions = []
    balanced_labels = []
    
    for idx in human_indices:
        balanced_sessions.append(human_sessions[idx])
        balanced_labels.append(0)
    
    for idx in bot_indices:
        balanced_sessions.append(bot_sessions[idx])
        balanced_labels.append(1)
    
    # Shuffle combined dataset
    combined_indices = np.random.permutation(len(balanced_sessions))
    all_sessions = [balanced_sessions[i] for i in combined_indices]
    all_labels = [balanced_labels[i] for i in combined_indices]
    
    print(f"Total samples: {len(all_sessions)}")
    print(f"  - Human: {sum(1 for l in all_labels if l == 0)}")
    print(f"  - Bot: {sum(1 for l in all_labels if l == 1)}")
    
    # =========================================================
    # PHASE 4: Train/Test Split
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 4: SPLITTING DATA")
    print("=" * 70)
    
    split_idx = int(len(all_sessions) * 0.8)
    
    train_sessions = all_sessions[:split_idx]
    train_labels = all_labels[:split_idx]
    test_sessions = all_sessions[split_idx:]
    test_labels = all_labels[split_idx:]
    
    print(f"Training set: {len(train_sessions)} samples")
    print(f"Test set: {len(test_sessions)} samples")
    
    # =========================================================
    # PHASE 5: Feature Extraction
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 5: EXTRACTING FEATURES")
    print("=" * 70)
    
    extractor = AdvancedFeatureExtractor()
    
    # Convert session format to expected format
    def convert_session(session):
        """Convert session dict to feature extractor format"""
        return {
            'mouse_data': session.get('mouse_movements', []),
            'keyboard_data': session.get('keystrokes', []),
            'click_data': session.get('clicks', []),
            'scroll_data': session.get('scrolls', [])
        }
    
    print("Extracting training features...")
    X_train = np.array([extractor.extract(convert_session(s)) for s in train_sessions])
    y_train = np.array(train_labels)
    
    print("Extracting test features...")
    X_test = np.array([extractor.extract(convert_session(s)) for s in test_sessions])
    y_test = np.array(test_labels)
    
    print(f"Feature shape: {X_train.shape}")
    
    # =========================================================
    # PHASE 6: Data Augmentation
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 6: DATA AUGMENTATION")
    print("=" * 70)
    
    from adversarial_training import FeatureAugmentor
    
    augmentor = FeatureAugmentor()
    X_train_aug, y_train_aug = augmentor.augment_batch(
        X_train, y_train, augmentation_factor=3
    )
    
    print(f"Augmented: {len(X_train)} → {len(X_train_aug)} samples")
    
    # =========================================================
    # PHASE 7: Adversarial Training
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 7: ADVERSARIAL TRAINING")
    print("=" * 70)
    
    # Normalize features
    feature_mean = X_train_aug.mean(axis=0)
    feature_std = X_train_aug.std(axis=0) + 1e-8
    X_train_norm = (X_train_aug - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std
    
    # Create model
    from ml_neural_network import BotDetectorNet
    from adversarial_training import AdversarialTrainer
    
    model = BotDetectorNet(input_size=X_train_norm.shape[1])
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train_norm)
    y_tensor = torch.LongTensor(y_train_aug.astype(int))
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    # Adversarial trainer
    adv_trainer = AdversarialTrainer(
        model,
        epsilon=0.1,
        attack_steps=10,
        attack_lr=0.01
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting adversarial training (with curriculum)...")
    history = adv_trainer.train_adversarial(
        train_loader,
        optimizer,
        epochs=80,
        attack_ratio=0.4,  # 40% adversarial examples
        curriculum=True,
        verbose=True
    )
    
    # =========================================================
    # PHASE 8: Evaluation
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 8: EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)
    
    # Test on clean examples
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_norm)
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_preds = (test_outputs > 0.5).numpy().astype(int)
    
    # Calculate metrics
    tp = np.sum((test_preds == 1) & (y_test == 1))
    fp = np.sum((test_preds == 1) & (y_test == 0))
    tn = np.sum((test_preds == 0) & (y_test == 0))
    fn = np.sum((test_preds == 0) & (y_test == 1))
    
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n{'=' * 50}")
    print(f"CLEAN TEST RESULTS")
    print(f"{'=' * 50}")
    print(f"Accuracy:           {accuracy:.1%}")
    print(f"Precision:          {precision:.1%}")
    print(f"Recall:             {recall:.1%}")
    print(f"F1 Score:           {f1:.3f}")
    print(f"False Positive Rate: {fpr:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    # =========================================================
    # PHASE 9: Adversarial Robustness Test
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 9: ADVERSARIAL ROBUSTNESS TEST")
    print("=" * 70)
    
    from adversarial_training import test_adversarial_robustness
    
    robustness = test_adversarial_robustness(
        model, X_test_norm, y_test,
        epsilon_values=[0.01, 0.05, 0.1, 0.15, 0.2]
    )
    
    # =========================================================
    # PHASE 10: Save Model
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 10: SAVING MODEL")
    print("=" * 70)
    
    os.makedirs("models", exist_ok=True)
    
    # Save PyTorch model
    model_path = "models/robust_bot_detector.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "input_size": X_train_norm.shape[1],
        "training_info": {
            "total_samples": len(X_train_aug),
            "attack_types": list(BOT_PROFILES.keys()),
            "adversarial_ratio": 0.4,
            "epochs": 80,
            "clean_accuracy": accuracy,
            "f1_score": f1,
            "robustness": robustness
        }
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Also save as the default model
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "input_size": X_train_norm.shape[1]
    }, "models/neural_bot_detector.pth")
    
    print("Also saved as: models/neural_bot_detector.pth")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("   TRAINING COMPLETE!")
    print("=" * 70)
    print(f"""
    Dataset:
      - Real human samples: {len(human_sessions)} (Balabit + CMU)
      - Bot attack types: {len(BOT_PROFILES) + 1} (including adversarial)
      - Total training samples: {len(X_train_aug)} (after augmentation)
    
    Model Performance:
      - Clean Accuracy: {accuracy:.1%}
      - F1 Score: {f1:.3f}
      - False Positive Rate: {fpr:.1%}
    
    Adversarial Robustness:
      - ε=0.01: {robustness.get(0.01, 'N/A'):.1%}
      - ε=0.05: {robustness.get(0.05, 'N/A'):.1%}
      - ε=0.10: {robustness.get(0.1, 'N/A'):.1%}
    
    Attack Types Trained On:
      {', '.join(BOT_PROFILES.keys())}
      + adversarial (GAN-like human mimicry)
    
    Files Saved:
      - models/robust_bot_detector.pth (full model + metadata)
      - models/neural_bot_detector.pth (production model)
    """)
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "robustness": robustness
    }


if __name__ == "__main__":
    results = main()
