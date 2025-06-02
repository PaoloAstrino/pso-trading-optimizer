import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from tqdm import tqdm
import os
import torch
from torch import nn
from matplotlib.widgets import Slider

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# DATA LOADING AND PREPROCESSING
# =====================================================================

def load_stock_data(file_path='nasdaq100_2012_2020.csv'):
    """Load stock data from CSV file and calculate daily returns"""
    # Load historical price data with proper date parsing
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        if df.empty:
            raise ValueError(f"No data found in {file_path}")
            
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        # Verify we have numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in the data")
        
        df = df[numeric_cols]  # Keep only numeric columns
        
        # Calculate daily returns using pct_change with explicit fill_method=None
        returns = df.pct_change(fill_method=None)
        
        # Drop any rows with all NaN values
        returns = returns.dropna(how='all')
        
        # Convert returns back to prices starting from base price of 100
        base_price = 100
        prices = pd.DataFrame(index=returns.index)
        
        for column in returns.columns:
            # Initialize the first price as base_price
            prices[column] = base_price * (1 + returns[column]).cumprod()
        
        return prices
        
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

class TradingRulesGPU:
    """GPU-accelerated trading rules implementation - simplified for toy model"""
    
    def __init__(self):
        # Expanded windows for more rules
        self.ma_short_windows = torch.tensor(range(10, 60, 5), device=device)  # Short windows from 10 to 55
        self.ma_long_windows = torch.tensor(range(60, 110, 5), device=device)  # Long windows from 60 to 105
        self.trb_windows = torch.tensor(range(5, 105, 5), device=device)  # TRB windows from 5 to 100
        self.rule_names = []
        self._generate_rule_names()
    
    def _generate_rule_names(self):
        """Generate names for all trading rules"""
        for s in self.ma_short_windows.cpu().numpy():
            for l in self.ma_long_windows.cpu().numpy():
                if s < l:
                    self.rule_names.append(f"MA_{s}_{l}")
        for w in self.trb_windows.cpu().numpy():
            self.rule_names.append(f"TRB_{w}")
    
    def generate_signals(self, prices):
        """Generate all trading signals using GPU acceleration"""
        prices_tensor = torch.tensor(prices.values, dtype=torch.float32, device=device)
        signals = []
        
        # MA signals
        for s in self.ma_short_windows:
            short_ma = self._moving_average_gpu(prices_tensor, s)
            for l in self.ma_long_windows:
                if s < l:
                    long_ma = self._moving_average_gpu(prices_tensor, l)
                    signal = (short_ma > long_ma).float() - (short_ma < long_ma).float()
                    signals.append(signal.flatten())
        
        # TRB signals
        for w in self.trb_windows:
            high = self._rolling_max_gpu(prices_tensor, w)
            low = self._rolling_min_gpu(prices_tensor, w)
            signal = ((prices_tensor > high).float() - (prices_tensor < low).float())
            signals.append(signal.flatten())
        
        # Stack signals and reshape to match expected dimensions
        stacked_signals = torch.stack(signals)
        return stacked_signals.reshape(len(signals), -1)
    
    def _moving_average_gpu(self, data, window):
        """Compute moving average using GPU"""
        kernel = torch.ones(window, device=device) / window
        # Ensure contiguous memory layout and proper reshaping
        data_reshaped = data.contiguous().reshape(1, 1, -1)
        kernel_reshaped = kernel.reshape(1, 1, -1)
        padding_size = (window - 1,)  # Specify padding as a tuple for 1D convolution
        return torch.conv1d(data_reshaped, kernel_reshaped, padding=padding_size)[:, :, window-1:].squeeze()
    
    def _rolling_max_gpu(self, data, window):
        """Compute rolling maximum using GPU"""
        # Reshape input for padding
        data_padded = torch.nn.functional.pad(data.contiguous(), (window-1, 0), mode='replicate')
        # Use unfold to create sliding windows and compute max
        windows = data_padded.unfold(dimension=1, size=window, step=1)
        return torch.max(windows, dim=2)[0]
    
    def _rolling_min_gpu(self, data, window):
        """Compute rolling minimum using GPU"""
        # Reshape input for padding
        data_padded = torch.nn.functional.pad(data.contiguous(), (window-1, 0), mode='replicate')
        # Use unfold to create sliding windows and compute min
        windows = data_padded.unfold(dimension=1, size=window, step=1)
        return torch.min(windows, dim=2)[0]

class PSOOptimizerGPU:
    """GPU-accelerated Particle Swarm Optimization - Toy Model with 3 particles and 5 iterations"""
    
    def __init__(self, num_particles=3, max_iterations=5, inertia_weight=0.9,
                 c1=2.5, c2=0.5, inertia_decay=0.4, w_min=0.4):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.inertia_decay = inertia_decay
        self.w_min = w_min
        
        # Calculate dimensions based on trading rules
        trading_rules = TradingRulesGPU()
        self.dimensions = len(trading_rules.rule_names) + 5  # rule weights + 5 parameters
        
        # Initialize particles on GPU
        self.particles = torch.rand(num_particles, self.dimensions, device=device)
        self.velocities = torch.zeros_like(self.particles)
        self.personal_best_positions = self.particles.clone()
        self.personal_best_fitness = torch.full((num_particles,), float('-inf'), device=device)
        # Initialize global best position with a random particle position
        self.global_best_position = self.particles[0].clone()
        self.global_best_fitness = float('-inf')
        
        # Store history for visualization
        self.particle_history = []
        self.fitness_history = []
        self.global_best_history = []
        self.global_best_fitness_history = []
        self.equity_history = None  # Store equity history for visualization
        
        # Initialize constraints
        self._apply_constraints()
    
    def _apply_constraints(self):
        """Apply constraints to particle positions"""
        # Get number of rules
        trading_rules = TradingRulesGPU()
        num_rules = len(trading_rules.rule_names)
        
        # Ensure non-negative weights and normalize
        self.particles[:, :num_rules] = torch.abs(self.particles[:, :num_rules])
        weights_sum = self.particles[:, :num_rules].sum(dim=1, keepdim=True)
        self.particles[:, :num_rules] = self.particles[:, :num_rules] / weights_sum
        
        # Clip other parameters
        self.particles[:, num_rules:num_rules+2].clamp_(0, 0.9)  # Thresholds
        self.particles[:, num_rules+2].clamp_(150, 300)  # Memory span
        self.particles[:, num_rules+3].clamp_(20, 150)  # Review span
        self.particles[:, num_rules+4].clamp_(0.1, 0.9)  # Reward factor
    
    def evaluate_fitness_batch(self, particles, stock_data, training_period):
        """Evaluate fitness for multiple particles in parallel"""
        # Get number of rules
        trading_rules = TradingRulesGPU()
        num_rules = len(trading_rules.rule_names)
        
        # Convert parameters to appropriate types
        rule_weights = particles[:, :num_rules]
        buy_threshold = particles[:, num_rules]
        sell_threshold = particles[:, num_rules+1]
        memory_span = particles[:, num_rules+2].int()
        review_span = particles[:, num_rules+3].int()
        reward_factor = particles[:, num_rules+4]
        
        # Generate trading signals
        signals = trading_rules.generate_signals(stock_data)
        
        # Reshape signals for batch processing
        batch_size = rule_weights.shape[0]
        num_days = stock_data.shape[0]
        num_stocks = stock_data.shape[1]
        signals = signals.reshape(signals.shape[0], -1)  # Flatten the time and stock dimensions
        
        # Compute weighted signals for each particle
        weighted_signals = torch.matmul(rule_weights, signals)
        
        # Reshape weighted_signals to match batch_size x num_days x num_stocks dimensions
        weighted_signals = weighted_signals.reshape(batch_size, num_days, num_stocks)
        
        # Calculate profitability for each rule across all time steps and stocks
        signals_reshaped = signals.reshape(num_rules, num_days, num_stocks)
        profitable_mask = (signals_reshaped > 0).float()
        N_plus = profitable_mask.sum(dim=(1, 2))  # Sum across days and stocks for each rule
        N = torch.tensor(num_days * num_stocks, device=device)
        
        # Calculate capped weight adjustment factor (h5/N)
        h5 = 0.1  # Hyperparameter for weight adjustment
        cap_factor = h5 / N
        
        # Calculate weight adjustments for each particle and rule
        weight_adjustments = torch.where(profitable_mask.sum(dim=(1, 2)) > 0,
                                       cap_factor * N / (N_plus + N),  # Reward for profitable rules
                                       -cap_factor * N_plus / (N_plus + N))  # Penalty for non-profitable rules
        
        # Apply weight adjustments to each particle's rule weights
        rule_weights += weight_adjustments.unsqueeze(0)
        
        # Ensure non-negative weights and normalize to maintain sum = 1
        rule_weights = torch.abs(rule_weights)
        weights_sum = rule_weights.sum(dim=1, keepdim=True)
        rule_weights = rule_weights / weights_sum
        
        # Apply memory span to smooth signals (exponential moving average)
        # Convert to float for division
        alpha = 2.0 / (memory_span.float() + 1.0).unsqueeze(1)
        smoothed_signals = weighted_signals.clone()
        
        # Generate trading decisions with memory effect
        # Reshape thresholds for proper broadcasting across all time steps and stocks
        buy_threshold_reshaped = buy_threshold.unsqueeze(1).unsqueeze(1).expand(-1, num_days, num_stocks)
        sell_threshold_reshaped = sell_threshold.unsqueeze(1).unsqueeze(1).expand(-1, num_days, num_stocks)
        
        # Compare signals with thresholds
        buy_signals = (smoothed_signals > buy_threshold_reshaped).float()
        sell_signals = (smoothed_signals < -sell_threshold_reshaped).float()
        
        # Get stock prices tensor
        prices_tensor = torch.tensor(stock_data.values, dtype=torch.float32, device=device).reshape(-1)
        
        # Set transaction cost (0.1% as specified)
        transaction_cost = 0.001
        
        # Initial equity for each stock ($100,000 as specified)
        initial_equity = 100000.0
        
        # Reshape buy/sell signals to match stock data dimensions for easier processing
        buy_signals_reshaped = buy_signals.reshape(batch_size, num_days * num_stocks)
        sell_signals_reshaped = sell_signals.reshape(batch_size, num_days * num_stocks)
        
        # Initialize equity tensor for each particle and stock
        # Shape: [batch_size, num_stocks]
        equity = torch.full((batch_size, num_stocks), initial_equity, device=device)
        
        # Reshape prices for easier access
        prices_reshaped = prices_tensor.reshape(num_days, num_stocks)
        
        # Track positions for each particle and stock (1 = in market, 0 = out of market)
        in_market = torch.zeros((batch_size, num_stocks), device=device)
        
        # Track buy prices for each position
        buy_prices = torch.zeros((batch_size, num_stocks), device=device)
        
        # Initialize equity history tensor for tracking daily equity changes
        self.daily_equity_history = torch.zeros((num_days, num_stocks), device=device)
        
        # Process each day sequentially to track equity changes
        for day in range(num_days):
            for stock_idx in range(num_stocks):
                # Get current price
                current_price = prices_reshaped[day, stock_idx]
                
                # Get signal indices
                signal_idx = day * num_stocks + stock_idx
                
                # Process sell signals first with sanity checks
                sell_mask = (sell_signals_reshaped[:, signal_idx] > 0) & (in_market[:, stock_idx] > 0)
                if torch.any(sell_mask):
                    # Prevent division by near-zero values
                    buy_prices[sell_mask, stock_idx] = torch.clamp(buy_prices[sell_mask, stock_idx], min=0.001)
                    
                    # Calculate selling price with transaction cost
                    sell_value = current_price * (1 - transaction_cost)
                    
                    # Calculate return for this trade with bounds
                    trade_return = torch.clamp(sell_value / (buy_prices[sell_mask, stock_idx] * (1 + transaction_cost)), max=2.0)
                    
                    # Update equity with bounded returns
                    equity[sell_mask, stock_idx] *= trade_return
                    
                    # Apply maximum equity constraint
                    max_allowed_equity = initial_equity * 10
                    equity[sell_mask, stock_idx] = torch.clamp(equity[sell_mask, stock_idx], max=max_allowed_equity)
                    
                    # Update market status
                    in_market[sell_mask, stock_idx] = 0
                
                # Process buy signals
                # Process buy signals with position sizing limits
                buy_mask = (buy_signals_reshaped[:, signal_idx] > 0) & (in_market[:, stock_idx] == 0)
                if torch.any(buy_mask):
                    # Limit position size to 10% of total equity per stock
                    max_position_size = equity[buy_mask].sum(dim=1, keepdim=True) * 0.1
                    position_size = torch.min(equity[buy_mask, stock_idx], max_position_size.squeeze())
                    
                    # Record buy price
                    buy_prices[buy_mask, stock_idx] = current_price
                    
                    # Update market status
                    in_market[buy_mask, stock_idx] = 1
                
                # Update daily equity considering both realized and unrealized gains/losses
                if day > 0:  # Copy previous day's equity as starting point
                    self.daily_equity_history[day, stock_idx] = self.daily_equity_history[day-1, stock_idx]
                else:  # First day starts with initial equity
                    self.daily_equity_history[day, stock_idx] = initial_equity
                
                # Track equity for the best particle (index 0) with realistic constraints
                if in_market[0, stock_idx] > 0:
                    # Calculate position value with limited unrealized returns
                    unrealized_return = torch.clamp(current_price / buy_prices[0, stock_idx], max=1.5)  # Limit max daily return to 50%
                    position_value = initial_equity * unrealized_return
                    
                    # Apply maximum equity constraint (10x initial)
                    max_allowed_equity = initial_equity * 10
                    position_value = torch.clamp(position_value, max=max_allowed_equity)
                    
                    # Update daily equity to reflect current position value
                    self.daily_equity_history[day, stock_idx] = position_value
        
        # Sell any remaining positions on the last day
        for stock_idx in range(num_stocks):
            last_price = prices_reshaped[-1, stock_idx]
            final_sell_mask = in_market[:, stock_idx] > 0
            
            if torch.any(final_sell_mask):
                # Calculate selling price with transaction cost
                sell_value = last_price * (1 - transaction_cost)
                
                # Calculate return for this final trade
                trade_return = sell_value / (buy_prices[final_sell_mask, stock_idx] * (1 + transaction_cost))
                
                # Update equity
                equity[final_sell_mask, stock_idx] *= trade_return
                
                # Update final day equity
                self.daily_equity_history[-1, stock_idx] = equity[0, stock_idx]
        
        # Calculate total equity across all stocks for each particle
        total_equity = torch.sum(equity, dim=1)
        
        # Always store the current equity history to track portfolio value changes
        self.equity_history = self.daily_equity_history.clone().cpu().numpy()
        
        # Calculate total return over the period
        total_return = total_equity / (initial_equity * num_stocks) - 1.0
        
        # Clip total return to reasonable values for a toy model
        total_return = torch.clamp(total_return, min=-0.9, max=5.0)
        
        # Annualize the return (assuming 252 trading days per year)
        num_years = num_days / 252.0
        # Use a more stable calculation for annualization with small time periods
        annual_net_profit = (1.0 + total_return) ** (1.0 / max(num_years, 0.1)) - 1.0
        
        # Ensure reasonable range for annual net profit in a toy model
        annual_net_profit = torch.clamp(annual_net_profit, min=-0.5, max=1.0)
        
        # Handle any NaN or inf values
        annual_net_profit = torch.nan_to_num(annual_net_profit, nan=-0.5, posinf=1.0, neginf=-0.5)
        
        return annual_net_profit
    
    def optimize(self, stock_data, training_period):
        """Run PSO optimization using GPU acceleration and store history for visualization"""
        
        # Get number of rules at the start
        trading_rules = TradingRulesGPU()
        num_rules = len(trading_rules.rule_names)
        
        progress_bar = tqdm(total=self.max_iterations, desc="GPU PSO Toy Model Optimization")
        
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all particles in parallel
            fitness = self.evaluate_fitness_batch(self.particles, stock_data, training_period)
            
            # Store history for visualization
            self.particle_history.append(self.particles.clone().cpu().numpy())
            self.fitness_history.append(fitness.clone().cpu().numpy())
            self.global_best_history.append(self.global_best_position.clone().cpu().numpy())
            self.global_best_fitness_history.append(self.global_best_fitness)
            
            # Filter out invalid fitness values (NaN, inf, -inf)
            valid_fitness = torch.isfinite(fitness)
            
            # Update personal best only for particles with valid fitness
            improved_mask = torch.logical_and(valid_fitness, fitness > self.personal_best_fitness)
            self.personal_best_positions[improved_mask] = self.particles[improved_mask]
            self.personal_best_fitness[improved_mask] = fitness[improved_mask]
            
            # Update global best based on new personal bests
            if torch.any(improved_mask):
                best_fitness = torch.max(self.personal_best_fitness[improved_mask])
                best_particle_idx = torch.argmax(self.personal_best_fitness[improved_mask]).item()
                
                # Update global best if this fitness is better
                if best_fitness > self.global_best_fitness:
                    self.global_best_fitness = best_fitness
                    self.global_best_position = self.personal_best_positions[best_particle_idx].clone()
        
            # Update inertia weight
            w = self.inertia_weight - (self.inertia_weight - self.w_min) * iteration / self.max_iterations
            
            # Update c1 and c2 values
            c1 = 2 - (1.5 * iteration / self.max_iterations)
            c2 = 0.5 + (1.5 * iteration / self.max_iterations)
            
            # Update velocities and positions
            r1 = torch.rand(self.num_particles, self.dimensions, device=device)
            r2 = torch.rand(self.num_particles, self.dimensions, device=device)
            
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.personal_best_positions - self.particles) +
                               c2 * r2 * (self.global_best_position - self.particles))
            
            # Update positions and apply constraints
            self.particles += self.velocities
            self._apply_constraints()
            
            # Ensure weights stay within historical bounds
            if len(self.particle_history) > 0:
                # Convert particle history to tensor format for weight bounds calculation
                historical_weights = torch.tensor(np.array([p[:, :num_rules] for p in self.particle_history]), device=device)
                min_weights = torch.min(historical_weights.view(-1, num_rules), dim=0)[0]
                max_weights = torch.max(historical_weights.view(-1, num_rules), dim=0)[0]
                self.particles[:, :num_rules] = torch.clamp(self.particles[:, :num_rules], min_weights, max_weights)
            
            progress_bar.update(1)
        
        # Store final iteration history
        self.particle_history.append(self.particles.clone().cpu().numpy())
        self.fitness_history.append(fitness.clone().cpu().numpy())
        self.global_best_history.append(self.global_best_position.clone().cpu().numpy())
        self.global_best_fitness_history.append(self.global_best_fitness)
        
        progress_bar.close()
            
        # Store training equity history
        self.training_equity_history = self.equity_history
        
        # Return best position and fitness
        return self.global_best_position, self.global_best_fitness
    
    def evaluate_test_performance(self, position, test_data, test_period):
        """Evaluate model performance on test data"""
        # Store training equity history before test evaluation
        training_equity_history = self.equity_history
        
        # Reset equity history for test period
        self.equity_history = None
        
        # Evaluate on test data
        test_fitness = self.evaluate_fitness_batch(position.unsqueeze(0), test_data, test_period)[0]
        
        # Store test equity history
        test_equity_history = self.equity_history
        
        # Plot training and test performance
        plt.figure(figsize=(12, 8))
        
        # Plot training equity curve
        plt.subplot(2, 1, 1)
        plt.plot(training_equity_history.mean(axis=1), label='Training Performance')
        plt.title('Portfolio Performance')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # Plot test equity curve
        plt.subplot(2, 1, 2)
        plt.plot(test_equity_history.mean(axis=1), label='Test Performance', color='orange')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        train_return = (training_equity_history[-1].mean() / training_equity_history[0].mean() - 1) * 100
        test_return = (test_equity_history[-1].mean() / test_equity_history[0].mean() - 1) * 100
        
        print(f'\nPerformance Metrics:')
        print(f'Training Return: {train_return:.2f}%')
        print(f'Test Return: {test_return:.2f}%')
        
        # Restore training equity history
        self.equity_history = training_equity_history
        
        # Store test equity history separately
        self.test_equity_history = test_equity_history
        
        return test_fitness

    def plot_particle_movement(self):
        """Visualize particle movement in 2D space with selectable trading rules"""
        # Check if particle history exists
        if not self.particle_history:
            print("No particle history data available for visualization")
            return
            
        # Get trading rules
        trading_rules = TradingRulesGPU()
        num_rules = len(trading_rules.rule_names)
        
        # Get the number of iterations and particles
        num_iterations = len(self.particle_history)
        
        # Create figure and axis with space for sliders
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot2grid((8, 1), (0, 0), rowspan=6)
        
        # Create sliders for rule selection
        ax_x = plt.subplot2grid((8, 1), (6, 0))
        ax_y = plt.subplot2grid((8, 1), (7, 0))
        
        slider_x = Slider(ax_x, 'X-axis Rule', 0, num_rules-1, valinit=0, valstep=1)
        slider_y = Slider(ax_y, 'Y-axis Rule', 0, num_rules-1, valinit=1, valstep=1)
        
        # Create a color map that transitions from white to vibrant colors
        base_colors = plt.cm.viridis(np.linspace(0, 1, num_iterations))
        colors = np.zeros((num_iterations, 4))
        colors[:, 3] = np.linspace(0.2, 1, num_iterations)  # Alpha channel
        colors[:, 0:3] = base_colors[:, 0:3]  # RGB channels
        
        # Create white-to-color transition
        for i in range(num_iterations):
            ratio = i / (num_iterations - 1)
            colors[i, 0:3] = (1 - ratio) * np.array([1, 1, 1]) + ratio * base_colors[i, 0:3]
        
        def update(val):
            ax.clear()
            x_rule = int(slider_x.val)
            y_rule = int(slider_y.val)
            
            # Extract selected dimensions for each particle across all iterations
            for i in range(self.num_particles):
                trajectory_x = []
                trajectory_y = []
                
                for iteration in range(num_iterations):
                    particle_pos = self.particle_history[iteration]
                    trajectory_x.append(particle_pos[i, x_rule])
                    trajectory_y.append(particle_pos[i, y_rule])
                    
                    # Plot each point with increasing color intensity
                    ax.scatter(particle_pos[i, x_rule], particle_pos[i, y_rule],
                              color=[colors[iteration]], alpha=colors[iteration, 3], s=50,
                              edgecolors='black', linewidth=0.5)
                
                # Plot the trajectory line
                ax.plot(trajectory_x, trajectory_y, alpha=0.3, color='gray')
            
            # Plot global best position
            ax.scatter(self.global_best_position[x_rule].cpu(), self.global_best_position[y_rule].cpu(),
                      c='red', marker='*', s=200, label='Global Best')
            
            ax.set_xlabel(f'Rule: {trading_rules.rule_names[x_rule]}')
            ax.set_ylabel(f'Rule: {trading_rules.rule_names[y_rule]}')
            ax.set_title('Particle Movement in Trading Rule Space')
            ax.legend()
            ax.grid(True)
            fig.canvas.draw_idle()
        
        # Connect sliders to update function
        slider_x.on_changed(update)
        slider_y.on_changed(update)
        
        # Initial plot
        update(None)
        
        plt.tight_layout()
        plt.show()

    def plot_initial_weights(self):
        """Plot the initial random weights for every trading rule"""
        trading_rules = TradingRulesGPU()
        num_rules = len(trading_rules.rule_names)
        
        # Extract initial weights for the first particle
        initial_weights = self.particles[0, :num_rules].cpu().numpy()
        
        # Calculate the average weight
        average_weight = np.mean(initial_weights)
        
        # Create a horizontal bar plot
        plt.figure(figsize=(8, 16))
        plt.barh(trading_rules.rule_names, initial_weights, color='red')
        plt.axvline(average_weight, color='grey', linestyle='--', linewidth=1, label=f'Average Weight: {average_weight:.2f}')
        plt.title('Initial Random Weights for Trading Rules')
        plt.ylabel('Trading Rules')
        plt.xlabel('Weight')
        plt.yticks(fontsize=5)  # Set font size for rule names
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_historical_weights(self):
        """Plot historical weights for each trading rule after optimization with clear min and max values"""
        trading_rules = TradingRulesGPU()
        num_rules = len(trading_rules.rule_names)
        
        # Extract rule weights from particle history
        historical_weights = []
        for particle in self.particle_history:
            # Extract weights for the best particle in each iteration
            if isinstance(particle, torch.Tensor):
                weights = particle[0, :num_rules].detach().cpu().numpy()
            else:
                weights = particle[0, :num_rules]
            historical_weights.append(weights)
        historical_weights = np.array(historical_weights)
        
        # Calculate min, max, and average weights for each rule
        min_weights = np.min(historical_weights, axis=0)
        max_weights = np.max(historical_weights, axis=0)
        avg_weights = np.mean(historical_weights, axis=0)
        final_weights = self.global_best_position[:num_rules].cpu().numpy()
        
        # Get rule names
        rule_names = trading_rules.rule_names[:len(min_weights)]
        
        # Sort rules by their final weights for better visualization
        sorted_indices = np.argsort(final_weights)[::-1]  # Descending order
        sorted_rule_names = [rule_names[i] for i in sorted_indices]
        sorted_min_weights = min_weights[sorted_indices]
        sorted_max_weights = max_weights[sorted_indices]
        sorted_avg_weights = avg_weights[sorted_indices]
        sorted_final_weights = final_weights[sorted_indices]
        
        # Create figure with adjusted dimensions
        height_per_rule = 0.15  # Adjust this value as needed
        plt.figure(figsize=(8, max(10, len(rule_names) * height_per_rule)))
        
        # Create y-axis positions for rules
        y_positions = np.arange(len(rule_names))
        
        # Plot min-max range as horizontal bars
        for i in range(len(sorted_min_weights)):
            plt.plot([sorted_min_weights[i], sorted_max_weights[i]], [y_positions[i], y_positions[i]], 
                    color='blue', linewidth=4, alpha=0.5)
            
            # Add min and max labels
            plt.text(sorted_min_weights[i], y_positions[i], f"{sorted_min_weights[i]:.3f}", 
                    va='center', ha='right', fontsize=5)
            plt.text(sorted_max_weights[i], y_positions[i], f"{sorted_max_weights[i]:.3f}", 
                    va='center', ha='left', fontsize=5)
        
        # Plot average weights as red dots
        plt.scatter(sorted_avg_weights, y_positions, color='red', s=10, label='Average Weight', zorder=2)
        
        # Plot final weights as green stars
        plt.scatter(sorted_final_weights, y_positions, color='yellow', marker='*', s=10, 
                    label='Final Optimized Weight', zorder=2)
        
        # Add mean weight line
        mean_weight = np.mean(sorted_avg_weights)
        plt.axvline(x=mean_weight, color='gray', linestyle='--', alpha=0.5, 
                    label=f'Mean Weight: {mean_weight:.3f}')
        
        # Set title and labels
        plt.title('Historical Trading Rule Weights (Min-Max Range)', fontsize=14)
        plt.xlabel('Weight', fontsize=12)
        plt.ylabel('Trading Rules', fontsize=12)
        
        # Use only rule_names for yticks
        plt.yticks(y_positions, sorted_rule_names, fontsize=5)
        
        # Add horizontal grid lines
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Set x-axis limits
        plt.xlim(0, max(sorted_max_weights) * 1.1)  # Add 10% margin
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Color code the rules by type
        ma_indices = [i for i, name in enumerate(sorted_rule_names) if name.startswith('MA_')]
        trb_indices = [i for i, name in enumerate(sorted_rule_names) if name.startswith('TRB_')]
        
        # Shade regions for MA and TRB rules
        if ma_indices:
            plt.axhspan(min(ma_indices)-0.5, max(ma_indices)+0.5, color='red', alpha=0.05, label='MA Rules')
        if trb_indices:
            plt.axhspan(min(trb_indices)-0.5, max(trb_indices)+0.5, color='blue', alpha=0.05, label='TRB Rules')
        
        plt.tight_layout()
        plt.show()
        
    def plot_trading_equity(self, stock_data, period_type='training'):
        """Plot trading equity curve over time with improved visualization and interactive zoom"""
        if self.equity_history is None:
            print(f"No equity history available for visualization for {period_type} period")
            return
            
        # Create figure with subplots to accommodate zoom control
        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_zoom = fig.add_subplot(gs[1])
        
        # Get dates from stock data index
        dates = stock_data.index
        
        # Select appropriate equity history based on period type
        if period_type == 'training':
            equity_data = self.training_equity_history
        else:  # test period
            equity_data = self.test_equity_history
            
        if equity_data is None:
            print(f"No equity data available for {period_type} period")
            return
            
        # Calculate total portfolio value over time
        total_portfolio_value = np.sum(equity_data, axis=1)
        
        # Ensure dimensions match
        if len(dates) != len(total_portfolio_value):
            print(f"Warning: Date dimensions ({len(dates)}) don't match equity data dimensions ({len(total_portfolio_value)})")
            return
        
        # Plot equity curve with gradient color on main axis
        ax_main.plot(dates, total_portfolio_value, 'b-', linewidth=2, label=f'{period_type.capitalize()} Portfolio Value')
        ax_main.fill_between(dates, total_portfolio_value, alpha=0.2)
        
        # Calculate and plot moving average
        window = max(5, len(total_portfolio_value) // 20)  # Adaptive window size
        ma = pd.Series(total_portfolio_value).rolling(window=window).mean()
        ax_main.plot(dates, ma, 'r--', alpha=0.7, label=f'{window}-day Moving Average')
        
        # Set y-axis limits with padding for better visibility
        value_range = total_portfolio_value.max() - total_portfolio_value.min()
        y_padding = value_range * 0.1
        ax_main.set_ylim(total_portfolio_value.min() - y_padding, total_portfolio_value.max() + y_padding)
        
        # Customize main plot
        ax_main.set_title(f'Portfolio Value Evolution - {period_type.capitalize()} Period', fontsize=14, pad=20)
        ax_main.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax_main.legend(loc='upper left')
        
        # Plot overview in zoom axis
        ax_zoom.plot(dates, total_portfolio_value, 'b-', linewidth=1)
        ax_zoom.fill_between(dates, total_portfolio_value, alpha=0.2)
        ax_zoom.set_xlabel('Time', fontsize=12)
        
        # Format zoom axis
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels
        for ax in [ax_main, ax_zoom]:
            ax.tick_params(axis='x', rotation=45)
        
        # Calculate and display key statistics
        initial_value = total_portfolio_value[0]
        final_value = total_portfolio_value[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        max_drawdown = np.min(total_portfolio_value / np.maximum.accumulate(total_portfolio_value) - 1) * 100
        
        stats_text = f'Total Return: {total_return:.1f}%\nMax Drawdown: {max_drawdown:.1f}%'
        ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8), fontsize=10,
                    verticalalignment='top')
        
        # Add zoom functionality
        def on_zoom_select(event):
            if event.button != 1:
                return
            if not ax_zoom.get_xlim(): return
            
            xmin, xmax = ax_zoom.get_xlim()
            ax_main.set_xlim(xmin, xmax)
            
            # Update y-axis limits based on visible data
            visible_data = total_portfolio_value[(dates >= pd.Timestamp(xmin)) & (dates <= pd.Timestamp(xmax))]
            if len(visible_data) > 0:
                value_range = visible_data.max() - visible_data.min()
                y_padding = value_range * 0.1
                ax_main.set_ylim(visible_data.min() - y_padding, visible_data.max() + y_padding)
            
            fig.canvas.draw_idle()
        
        # Connect the zoom functionality
        ax_zoom.callbacks.connect('xlim_changed', on_zoom_select)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
def main():
    """Main function to run the GPU-accelerated PSO toy model optimization"""
    # Load stock data
    stock_data = load_stock_data()
    print(f"Loaded data for {len(stock_data.columns)} stocks from {stock_data.index[0]} to {stock_data.index[-1]}")
    
    # Define training and test periods
    split_idx = int(len(stock_data) * 0.6)
    train_start = stock_data.index[0]
    train_end = stock_data.index[split_idx]
    test_start = stock_data.index[split_idx + 1]
    test_end = stock_data.index[-1]
    
    # Split data into train and test sets
    train_data = stock_data.loc[train_start:train_end]
    test_data = stock_data.loc[test_start:test_end]
    
    print(f"Training period: {train_start} to {train_end} ({len(train_data)} days)")
    print(f"Testing period: {test_start} to {test_end} ({len(test_data)} days)")
    print(f"Using device: {device}")
    
    # Initialize and run GPU-accelerated PSO with toy model parameters
    pso = PSOOptimizerGPU(num_particles=10, max_iterations=100)
    
    # Train the model
    print("\nTraining phase:")
    start_time = time.time()
    best_position, train_fitness = pso.optimize(train_data, (train_start, train_end))
    train_time = time.time() - start_time
    
    # Store training equity curve
    training_equity = pso.equity_history.copy() if pso.equity_history is not None else None
    
    # Test the model
    print("\nTesting phase:")
    test_fitness = pso.evaluate_test_performance(best_position, test_data, (test_start, test_end))
    test_equity = pso.equity_history.copy() if pso.equity_history is not None else None
    end_time = time.time()
    
    # Compare training and test equity curves
    print("\nEquity curves have been plotted for both training and test periods.")
    print("Please examine them to assess strategy performance and stability.")
    
    # Get number of rules
    trading_rules = TradingRulesGPU()
    num_rules = len(trading_rules.rule_names)
    
    # Extract parameters from the best position tensor
    buy_threshold = best_position[num_rules].item()
    sell_threshold = best_position[num_rules+1].item()
    memory_span = max(5, best_position[num_rules+2].int().item())  # Ensure minimum value of 5
    review_span = max(1, best_position[num_rules+3].int().item())  # Ensure minimum value of 1
    reward_factor = best_position[num_rules+4].item()
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds (Training: {train_time:.2f}s)")
    print(f"Training fitness (Annual Net Profit): {train_fitness:.4f} ({train_fitness * 100:.2f}%)")
    print(f"Testing fitness (Annual Net Profit): {test_fitness:.4f} ({test_fitness * 100:.2f}%)")
    # Calculate Sharpe ratio for both periods (assuming risk-free rate = 0 for simplicity)
    train_returns = np.diff(training_equity.mean(axis=1)) / training_equity.mean(axis=1)[:-1]
    test_returns = np.diff(test_equity.mean(axis=1)) / test_equity.mean(axis=1)[:-1]
    
    train_sharpe = np.sqrt(252) * (train_returns.mean() / train_returns.std())
    test_sharpe = np.sqrt(252) * (test_returns.mean() / test_returns.std())
    
    stability_ratio = test_fitness/train_fitness
    
    print(f"\nPerformance Metrics:")
    print(f"Training Sharpe Ratio: {train_sharpe:.2f}")
    print(f"Testing Sharpe Ratio: {test_sharpe:.2f}")
    print(f"Out-of-sample performance ratio: {stability_ratio:.2f}")
    
    # Determine stability category and explanation
    if stability_ratio > 0.8:
        stability = "Good"
        explanation = "The model shows good stability with consistent performance between training and testing periods. "
        explanation += f"The Sharpe ratios (train: {train_sharpe:.2f}, test: {test_sharpe:.2f}) indicate reliable risk-adjusted returns."
    elif stability_ratio > 0.5:
        stability = "Poor"
        explanation = "The model shows signs of overfitting with degraded performance in the test period. "
        explanation += f"The difference in Sharpe ratios (train: {train_sharpe:.2f}, test: {test_sharpe:.2f}) suggests inconsistent risk-adjusted returns."
    else:
        stability = "Bad"
        explanation = "The model shows severe overfitting with significant performance deterioration in the test period. "
        explanation += f"The large gap in Sharpe ratios (train: {train_sharpe:.2f}, test: {test_sharpe:.2f}) indicates unstable risk-adjusted returns."
    
    print(f"Model stability: {stability}")
    print(f"Explanation: {explanation}")

    print("\nOptimized Parameters:")
    print(f"Buy Threshold: {buy_threshold:.4f}")
    print(f"Sell Threshold: {sell_threshold:.4f}")
    print(f"Memory Span: {memory_span} days")
    print(f"Review Span: {review_span} days")
    print(f"Reward Factor: {reward_factor:.4f}")

    # Visualize particle movement
    print("\nVisualizing particle movement...")
    pso.plot_particle_movement()

    # Visualize historical weights
    print("\nVisualizing historical weights...")
    pso.plot_historical_weights()

    # Visualize trading equity curves
    print("\nVisualizing trading equity curves...")
    print("Training period equity curve:")
    pso.plot_trading_equity(train_data)
    print("\nTest period equity curve:")
    pso.plot_trading_equity(test_data)



if __name__ == '__main__':
    main()

