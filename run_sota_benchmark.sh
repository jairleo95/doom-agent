#!/bin/bash
# 🚀 Running SOTA Benchmark (100 Episodes per Agent)
# This may take ~45-60 minutes depending on hardware (Arnold is CPU-bound)

echo "Starting rigorous benchmark (N=100)..."
echo "Results will be saved to: results/benchmarks/"

# Run the scientific benchmark script
python3 scripts/run_scientific_benchmark.py --episodes 100

echo "✅ Benchmark Complete!"
echo "Check 'results/benchmarks/SCIENTIFIC_BENCHMARK.md' for the report."
