"""
Inference Testing Script for Ensemble Model
Randomly samples test data and measures inference times
"""
import numpy as np
import pandas as pd
from ensemble_voting import VotingEnsemble, preprocess_data
import time
import pickle
from pathlib import Path
import argparse
import json
from datetime import datetime

def run_inference_test(model_path, test_path, sample_size=1000, num_samples=5):
    """
    Run inference tests on randomly sampled test data

    Args:
        model_path: Path to the trained ensemble model
        test_path: Path to test data CSV
        sample_size: Number of samples to run inference on each test
        num_samples: Number of random samples to test
    """
    print(f"🚀 Loading ensemble model from: {model_path}")

    # Load ensemble model
    ensemble = VotingEnsemble.load(model_path)

    print(f"📂 Loading test data from: {test_path}")

    # Load test data
    test_df = pd.read_csv(test_path)

    print(f"📊 Test data shape: {test_df.shape}")
    print(f"📊 Fraud rate in test data: {test_df['is_fraud'].mean():.4f}")

    # Preprocess test data
    # We'll create a dummy train_df with same columns for preprocessing function
    dummy_train_df = test_df.sample(n=min(1000, len(test_df)), random_state=42)  # Use a sample for dummy
    X_full, _, y_full, _ = preprocess_data(dummy_train_df, test_df)

    print(f"✅ Preprocessed data shape: {X_full.shape}")

    # Store inference results
    inference_results = []

    print(f"\n🔬 Running {num_samples} inference tests with {sample_size} samples each...")

    for i in range(num_samples):
        print(f"\n📊 Test {i+1}/{num_samples}")

        # Randomly sample test data
        sample_indices = np.random.choice(X_full.shape[0], size=sample_size, replace=False)
        X_sample = X_full[sample_indices]
        y_sample = y_full[sample_indices]

        print(f"   Sample shape: {X_sample.shape}")
        print(f"   True fraud rate in sample: {y_sample.mean():.4f}")

        # Measure different aspects of inference time
        # 1. Overall inference time
        start_time = time.time()
        y_pred = ensemble.predict(X_sample, threshold=0.13)
        end_time = time.time()

        total_inference_time = end_time - start_time
        samples_per_second = sample_size / total_inference_time

        # 2. Per-sample inference time
        per_sample_time = total_inference_time / sample_size

        # 3. Predict probabilities time separately (if needed)
        start_proba_time = time.time()
        y_proba = ensemble.predict_proba(X_sample)
        end_proba_time = time.time()
        proba_time = end_proba_time - start_proba_time

        # Additional metrics
        true_positives = np.sum((y_sample == 1) & (y_pred == 1))
        false_positives = np.sum((y_sample == 0) & (y_pred == 1))
        true_negatives = np.sum((y_sample == 0) & (y_pred == 0))
        false_negatives = np.sum((y_sample == 1) & (y_pred == 1) == False)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"   Total inference time: {total_inference_time:.4f}s")
        print(f"   Time per sample: {per_sample_time:.6f}s ({per_sample_time*1000:.3f}ms)")
        print(f"   Samples per second: {samples_per_second:.2f}")
        print(f"   Probability calculation time: {proba_time:.4f}s")
        print(f"   True Positives: {true_positives}, False Positives: {false_positives}")
        print(f"   True Negatives: {true_negatives}, False Negatives: {false_negatives}")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Store results
        result = {
            'test_id': i+1,
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size,
            'total_inference_time': total_inference_time,
            'per_sample_time': per_sample_time,
            'per_sample_time_ms': per_sample_time * 1000,
            'samples_per_second': samples_per_second,
            'probability_calc_time': proba_time,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fraud_rate_in_sample': float(y_sample.mean()),
            'predicted_fraud_rate': float(y_pred.mean()),
            'true_positive_rate': float(recall),  # Same as recall
            'false_positive_rate': false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        }

        inference_results.append(result)

    return inference_results

def save_inference_results(results, output_dir='results/inference_tests'):
    """
    Save inference results to files

    Args:
        results: List of inference test results
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    json_path = Path(output_dir) / f"inference_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary CSV
    df = pd.DataFrame(results)
    csv_path = Path(output_dir) / f"inference_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Create summary statistics
    avg_inference_time = np.mean([r['total_inference_time'] for r in results])
    avg_samples_per_second = np.mean([r['samples_per_second'] for r in results])
    avg_per_sample_time = np.mean([r['per_sample_time'] for r in results])
    avg_per_sample_time_ms = np.mean([r['per_sample_time_ms'] for r in results])
    avg_proba_time = np.mean([r['probability_calc_time'] for r in results])
    avg_fraud_rate = np.mean([r['fraud_rate_in_sample'] for r in results])
    avg_predicted_fraud_rate = np.mean([r['predicted_fraud_rate'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])

    summary_stats = {
        'total_tests_run': len(results),
        'average_inference_time': avg_inference_time,
        'average_samples_per_second': avg_samples_per_second,
        'average_per_sample_time': avg_per_sample_time,
        'average_per_sample_time_ms': avg_per_sample_time_ms,
        'average_probability_calc_time': avg_proba_time,
        'average_fraud_rate_in_samples': avg_fraud_rate,
        'average_predicted_fraud_rate': avg_predicted_fraud_rate,
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1_score': avg_f1,
        'timestamp': datetime.now().isoformat()
    }

    stats_path = Path(output_dir) / f"inference_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)

    # Generate performance report
    report_path = Path(output_dir) / f"inference_performance_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("ADWC-DFS Inference Performance Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests run: {len(results)}\n")
        f.write(f"Sample size per test: {results[0]['sample_size']}\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Average Inference Time: {avg_inference_time:.4f}s\n")
        f.write(f"  Average Time per Sample: {avg_per_sample_time_ms:.3f}ms ({avg_per_sample_time:.6f}s)\n")
        f.write(f"  Average Samples/Second: {avg_samples_per_second:.2f}\n")
        f.write(f"  Average Probability Calc Time: {avg_proba_time:.4f}s\n\n")

        f.write("ACCURACY METRICS:\n")
        f.write(f"  Average Precision: {avg_precision:.4f}\n")
        f.write(f"  Average Recall: {avg_recall:.4f}\n")
        f.write(f"  Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"  Average Fraud Rate (True): {avg_fraud_rate:.4f}\n")
        f.write(f"  Average Fraud Rate (Predicted): {avg_predicted_fraud_rate:.4f}\n\n")

        f.write("DETAILED RESULTS PER TEST:\n")
        f.write("Test ID | Time(s) | Time/Sample(ms) | Samples/s | Precision | Recall | F1 Score\n")
        f.write("-" * 85 + "\n")
        for result in results:
            f.write(f"{result['test_id']:>7} | {result['total_inference_time']:>7.4f} | {result['per_sample_time_ms']:>13.3f} | {result['samples_per_second']:>9.2f} | {result['precision']:>9.4f} | {result['recall']:>6.4f} | {result['f1_score']:>8.4f}\n")

    # Generate a simple plot of inference performance if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        # Plot 1: Inference time per test
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        test_ids = [r['test_id'] for r in results]
        times = [r['total_inference_time'] for r in results]
        plt.plot(test_ids, times, marker='o', linewidth=2, markersize=8)
        plt.title('Total Inference Time per Test')
        plt.xlabel('Test ID')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)

        # Plot 2: Time per sample
        plt.subplot(2, 2, 2)
        per_sample_times = [r['per_sample_time_ms'] for r in results]
        plt.plot(test_ids, per_sample_times, marker='s', linewidth=2, markersize=8, color='orange')
        plt.title('Average Time per Sample')
        plt.xlabel('Test ID')
        plt.ylabel('Time (milliseconds)')
        plt.grid(True, alpha=0.3)

        # Plot 3: Performance (samples per second)
        plt.subplot(2, 2, 3)
        samples_per_sec = [r['samples_per_second'] for r in results]
        plt.plot(test_ids, samples_per_sec, marker='^', linewidth=2, markersize=8, color='green')
        plt.title('Throughput (Samples per Second)')
        plt.xlabel('Test ID')
        plt.ylabel('Samples/Second')
        plt.grid(True, alpha=0.3)

        # Plot 4: Accuracy metrics
        plt.subplot(2, 2, 4)
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        x = np.arange(len(results))
        width = 0.25

        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)

        plt.xlabel('Test ID')
        plt.ylabel('Score')
        plt.title('Accuracy Metrics per Test')
        plt.legend()
        plt.xticks(x, test_ids)
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = Path(output_dir) / f"inference_performance_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   - Performance plot: {plot_path.name}")
    except ImportError:
        print("   (Matplotlib not available, skipping performance plots)")

    print(f"\n💾 Results saved to {output_dir}/")
    print(f"   - Detailed results: {json_path.name}")
    print(f"   - Summary CSV: {csv_path.name}")
    print(f"   - Summary stats: {stats_path.name}")
    print(f"   - Performance report: {report_path.name}")

    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Average inference time: {avg_inference_time:.4f}s")
    print(f"   Average time per sample: {avg_per_sample_time_ms:.3f}ms")
    print(f"   Average samples/second: {avg_samples_per_second:.2f}")
    print(f"   Average precision: {avg_precision:.4f}")
    print(f"   Average recall: {avg_recall:.4f}")
    print(f"   Average F1 score: {avg_f1:.4f}")

    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Run inference tests on ensemble model')
    parser.add_argument('--model_path', default='results/ensemble_model.pkl',
                       help='Path to trained ensemble model')
    parser.add_argument('--test_path', default='data/test.csv',
                       help='Path to test CSV data')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of samples to test in each inference run')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of random samples to test')
    parser.add_argument('--output_dir', default='results/inference_tests',
                       help='Directory to save inference results')
    
    args = parser.parse_args()
    
    print("🚀 ADWC-DFS Inference Testing")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_path}")
    print(f"Sample size: {args.sample_size}")
    print(f"Number of tests: {args.num_samples}")
    print("=" * 80)
    
    # Run inference tests
    results = run_inference_test(
        model_path=args.model_path,
        test_path=args.test_path,
        sample_size=args.sample_size,
        num_samples=args.num_samples
    )
    
    # Save results
    save_inference_results(results, args.output_dir)
    
    print("\n✅ Inference testing complete!")

if __name__ == "__main__":
    main()