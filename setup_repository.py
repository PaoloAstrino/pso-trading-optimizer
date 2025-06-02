#!/usr/bin/env python3
"""
Setup script for PSO Trading Strategy Optimizer repository
This script helps organize the project structure and prepare it for GitHub
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the recommended directory structure"""
    
    base_dir = Path.cwd()
    
    # Create main directories
    directories = [
        'src',
        'data',
        'results',
        'docs',
        'tests',
        'notebooks',
        'src/utils'
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return base_dir

def organize_files(base_dir):
    """Move files to appropriate directories"""
    
    # Move source files
    source_files = [
        'exam_pso_gpu_toy.py',
        'nasdaq100_scraping.py'
    ]
    
    for file in source_files:
        if (base_dir / file).exists():
            shutil.move(str(base_dir / file), str(base_dir / 'src' / file))
            print(f"✅ Moved {file} to src/")
    
    # Move data files
    data_files = [
        'nasdaq100_2012_2020.csv'
    ]
    
    for file in data_files:
        if (base_dir / file).exists():
            shutil.move(str(base_dir / file), str(base_dir / 'data' / file))
            print(f"✅ Moved {file} to data/")
    
    # Move result files (images and analysis)
    result_files = [
        'equity_lines.png',
        'historycal_weights.png',
        'rules1.png',
        'rules2.png',
        'rules3.png',
        'rules4.png',
        'rules5.png',
        'rules6.png',
        'PRS with PSO.pdf'
    ]
    
    for file in result_files:
        if (base_dir / file).exists():
            shutil.move(str(base_dir / file), str(base_dir / 'results' / file))
            print(f"✅ Moved {file} to results/")
    
    # Move zip file to results if it exists
    if (base_dir / 'exam_pso_gpu_toy_Paolo_Astrino.zip').exists():
        shutil.move(str(base_dir / 'exam_pso_gpu_toy_Paolo_Astrino.zip'), 
                   str(base_dir / 'results' / 'exam_pso_gpu_toy_Paolo_Astrino.zip'))
        print("✅ Moved zip file to results/")

def create_init_files(base_dir):
    """Create __init__.py files for Python packages"""
    
    init_files = [
        'src/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = base_dir / init_file
        init_path.touch()
        print(f"✅ Created {init_file}")

def create_additional_docs(base_dir):
    """Create additional documentation files"""
    
    docs_dir = base_dir / 'docs'
    
    # Algorithm details
    algorithm_doc = docs_dir / 'algorithm_details.md'
    with open(algorithm_doc, 'w') as f:
        f.write("""# PSO Algorithm Details

## Particle Swarm Optimization Implementation

### Overview
This document provides detailed information about the PSO algorithm implementation used in the trading strategy optimizer.

### Parameters
- **Population Size**: 10 particles
- **Iterations**: 100 maximum iterations
- **Inertia Weight**: 0.9 (with decay)
- **Cognitive Parameter (c1)**: 2.5
- **Social Parameter (c2)**: 0.5

### Optimization Process
1. **Initialization**: Random particle positions within constraints
2. **Fitness Evaluation**: Parallel evaluation using GPU acceleration
3. **Position Update**: Based on personal and global best positions
4. **Constraint Handling**: Automatic bounds enforcement and normalization

### GPU Acceleration
- PyTorch tensor operations for parallel processing
- Batch evaluation of multiple particles simultaneously
- Efficient memory management for large-scale optimization

""")
    
    # Trading rules documentation
    trading_doc = docs_dir / 'trading_rules.md'
    with open(trading_doc, 'w') as f:
        f.write("""# Trading Rules Documentation

## Implemented Trading Strategies

### Moving Average (MA) Rules
- **Concept**: Trend-following strategy based on moving average crossovers
- **Implementation**: 50 different MA combinations
- **Parameters**: Short MA (10-55 days), Long MA (60-105 days)
- **Signals**: Buy when short MA > long MA, sell when short MA < long MA

### Trading Range Breakout (TRB) Rules
- **Concept**: Momentum strategy based on price breakouts
- **Implementation**: 20 different lookback periods
- **Parameters**: Lookback windows (5-100 days)
- **Signals**: Buy on breakout above highest high, sell on breakdown below lowest low

### Signal Processing
- **GPU Acceleration**: Parallel computation of all signals
- **Memory Effects**: Exponential smoothing with configurable memory span
- **Dynamic Weighting**: Real-time adjustment based on rule profitability

""")
    
    print("✅ Created additional documentation files")

def generate_git_commands():
    """Generate Git commands for repository setup"""
    
    commands = """
# Git Repository Setup Commands

# 1. Initialize Git repository
git init

# 2. Add all files
git add .

# 3. Create initial commit
git commit -m "Initial commit: GPU-accelerated PSO trading strategy optimizer"

# 4. Create GitHub repository (replace with your username)
# Go to GitHub and create a new repository named 'pso-trading-optimizer'

# 5. Add remote origin (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/pso-trading-optimizer.git

# 6. Push to GitHub
git branch -M main
git push -u origin main

# 7. Create development branch
git checkout -b development
git push -u origin development
"""
    
    with open('GIT_SETUP.md', 'w') as f:
        f.write(commands)
    
    print("✅ Created Git setup instructions: GIT_SETUP.md")

def main():
    """Main setup function"""
    print("🚀 Setting up PSO Trading Strategy Optimizer repository...\n")
    
    try:
        # Create directory structure
        base_dir = create_directory_structure()
        print()
        
        # Organize existing files
        organize_files(base_dir)
        print()
        
        # Create Python package files
        create_init_files(base_dir)
        print()
        
        # Create additional documentation
        create_additional_docs(base_dir)
        print()
        
        # Generate Git commands
        generate_git_commands()
        print()
        
        print("✅ Repository setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Review the organized file structure")
        print("2. Update any file paths in the code if necessary")
        print("3. Follow the Git setup instructions in GIT_SETUP.md")
        print("4. Create your GitHub repository")
        print("5. Push to GitHub and share your portfolio project!")
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
