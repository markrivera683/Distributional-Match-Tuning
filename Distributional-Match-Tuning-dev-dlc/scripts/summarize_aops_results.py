#!/usr/bin/env python3
"""
Summarize AoPS Triplet Experiment Results

Usage:
    python summarize_aops_results.py --results_dir /mnt/workspace/runs
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_tensorboard_event_file(event_file: str) -> Dict:
    """Extract key metrics from tensorboard event file."""
    metrics = {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Extract scalar metrics
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = {
                    'final': events[-1].value,
                    'max': max(e.value for e in events),
                    'min': min(e.value for e in events),
                }
    except Exception as e:
        print(f"Warning: Could not parse {event_file}: {e}")
    
    return metrics


def parse_train_log(log_file: str) -> Dict:
    """Extract key info from train log."""
    info = {
        'smoke_passed': False,
        'has_global_step': False,
        'has_eval': False,
        'has_errors': False,
        'critic_lr': None,
        'adapter_drift': None,
    }
    
    if not os.path.exists(log_file):
        return info
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Check smoke criteria
    info['has_global_step'] = 'train/global_step' in content
    info['has_eval'] = 'Evaluation completed' in content
    info['has_errors'] = any(err in content for err in [
        'FileNotFoundError', 'KeyError', 'RuntimeError', 'ActorDiedError'
    ])
    info['smoke_passed'] = info['has_global_step'] and info['has_eval'] and not info['has_errors']
    
    # Extract Group 3 specific metrics
    critic_lr_matches = re.findall(r'critic_opt_lr_group1[=:\s]+([\d.e+-]+)', content)
    if critic_lr_matches:
        info['critic_lr'] = float(critic_lr_matches[-1])
    
    drift_matches = re.findall(r'feature_adapter_drift[=:\s]+([\d.e+-]+)', content)
    if drift_matches:
        info['adapter_drift'] = float(drift_matches[-1])
    
    return info


def find_experiment_results(results_dir: str, group: int) -> Optional[Dict]:
    """Find and parse results for a specific group."""
    # Find run directory
    pattern = f"aops_g{group}_*"
    matching_dirs = list(Path(results_dir).glob(pattern))
    
    if not matching_dirs:
        return None
    
    # Use the most recent run
    run_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
    
    result = {
        'group': group,
        'run_name': run_dir.name,
        'run_path': str(run_dir),
    }
    
    # Parse train log
    log_file = run_dir / 'train.log'
    if log_file.exists():
        log_info = parse_train_log(str(log_file))
        result.update(log_info)
    
    # Parse tensorboard metrics
    tb_dir = run_dir / 'tensorboard'
    if tb_dir.exists():
        event_files = list(tb_dir.glob('events.out.tfevents.*'))
        if event_files:
            result['tensorboard'] = parse_tensorboard_event_file(str(event_files[0]))
    
    return result


def generate_summary_table(results: List[Dict]) -> str:
    """Generate a formatted summary table."""
    lines = []
    lines.append("=" * 100)
    lines.append("AoPS TRIPLET EXPERIMENT SUMMARY")
    lines.append("=" * 100)
    lines.append("")
    
    # Header
    lines.append(f"{'Group':<10} {'Name':<30} {'Smoke':<8} {'Global Step':<12} {'Eval':<8} {'Critic LR':<15} {'Adapter Drift':<15}")
    lines.append("-" * 100)
    
    for r in results:
        group = r.get('group', '-')
        name = r.get('run_name', '-')[:28]
        smoke = 'PASS' if r.get('smoke_passed') else 'FAIL'
        step = 'YES' if r.get('has_global_step') else 'NO'
        eval_ = 'YES' if r.get('has_eval') else 'NO'
        
        critic_lr = r.get('critic_lr')
        critic_lr_str = f"{critic_lr:.2e}" if critic_lr is not None else 'N/A'
        
        drift = r.get('adapter_drift')
        drift_str = f"{drift:.4f}" if drift is not None else 'N/A'
        
        lines.append(f"{group:<10} {name:<30} {smoke:<8} {step:<12} {eval_:<8} {critic_lr_str:<15} {drift_str:<15}")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 100)
    
    for r in results:
        lines.append("")
        lines.append(f"Group {r.get('group', '?')}: {r.get('run_name', 'Unknown')}")
        lines.append(f"  Path: {r.get('run_path', 'N/A')}")
        lines.append(f"  Smoke Passed: {r.get('smoke_passed', False)}")
        lines.append(f"  Has Global Step: {r.get('has_global_step', False)}")
        lines.append(f"  Has Evaluation: {r.get('has_eval', False)}")
        lines.append(f"  Has Errors: {r.get('has_errors', True)}")
        
        if r.get('critic_lr') is not None:
            lines.append(f"  Critic LR: {r['critic_lr']:.2e}")
        if r.get('adapter_drift') is not None:
            lines.append(f"  Adapter Drift: {r['adapter_drift']:.4f}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize AoPS Triplet Results')
    parser.add_argument('--results_dir', type=str, default='/mnt/workspace/runs',
                        help='Directory containing experiment results')
    args = parser.parse_args()
    
    print(f"Scanning results in: {args.results_dir}")
    
    # Collect results for all 3 groups
    results = []
    for group in [1, 2, 3]:
        result = find_experiment_results(args.results_dir, group)
        if result:
            results.append(result)
            print(f"  Found Group {group}: {result['run_name']}")
        else:
            print(f"  Group {group}: Not found")
    
    # Generate and save summary
    summary = generate_summary_table(results)
    
    summary_file = os.path.join(args.results_dir, 'triplet_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\n" + summary)
    
    # Also save as JSON
    json_file = os.path.join(args.results_dir, 'triplet_summary.json')
    with open(json_file, 'w') as f:
        json.dump({
            'results': results,
            'generated_at': str(Path().stat().st_mtime if Path().exists() else ''),
        }, f, indent=2, default=str)
    
    print(f"JSON saved to: {json_file}")


if __name__ == '__main__':
    main()
