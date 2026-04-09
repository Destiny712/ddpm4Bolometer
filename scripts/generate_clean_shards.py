"""
Generate clean pulse windows in shards.

Usage:
    python -u scripts/generate_clean_shards.py --output_dir /path/to/output
    python -u scripts/generate_clean_shards.py --output_dir /path/to/output --E_min 1 --E_max 150
    python -u scripts/generate_clean_shards.py --output_dir /path/to/output --E_min 150 --E_max 5407
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pulse.generate import generate_dataset

parser = argparse.ArgumentParser(description="Generate clean pulse shards")
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--n_shards', type=int, default=10)
parser.add_argument('--windows_per_shard', type=int, default=10000)
parser.add_argument('--base_seed', type=int, default=42)
parser.add_argument('--E_min', type=float, default=1.0)
parser.add_argument('--E_max', type=float, default=5407.0)
parser.add_argument('--pileup_fraction', type=float, default=0.3)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

for shard in range(args.n_shards):
    output_file = os.path.join(args.output_dir, f'clean_{shard:03d}.h5')
    if os.path.exists(output_file):
        print(f"Shard {shard} already exists, skipping: {output_file}")
        continue

    seed = args.base_seed + shard * 1000
    print(f"\n{'='*60}")
    print(f"Generating shard {shard}/{args.n_shards}: {output_file}")
    print(f"  Windows: {args.windows_per_shard}, seed: {seed}")
    print(f"  Energy range: [{args.E_min}, {args.E_max}] keV")
    print(f"{'='*60}")

    generate_dataset(
        n_pulses=args.windows_per_shard,
        output_file=output_file,
        duration=10.0,
        f_sample=1000.0,
        pileup_fraction=args.pileup_fraction,
        seed=seed,
        E_min=args.E_min,
        E_max=args.E_max,
    )

print(f"\nAll {args.n_shards} shards complete.")
