import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_visualizations(arima_df, lstm_df, fixed_df):
    # Align data lengths first
    min_length = min(len(arima_df), len(lstm_df), len(fixed_df))
    arima_df = arima_df.iloc[:min_length]
    lstm_df = lstm_df.iloc[:min_length]
    fixed_df = fixed_df.iloc[:min_length]
    
    plt.style.use('bmh')
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Traffic Control System Comparison', fontsize=16, y=0.95)
    
    colors = {'arima': '#590942', 'lstm': '#3498db', 'fixed': '#e74c3c'}
    line_styles = {'arima': '-', 'lstm': '--', 'fixed': ':'}
    
    def style_axis(ax, title, xlabel, ylabel):
        ax.set_title(title, pad=20, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
    
    window = 30
    
    # Waiting Time Plot
    axes[0, 0].plot(arima_df.index, arima_df['waiting_time'].rolling(window).mean(), 
                      color=colors['arima'], linestyle=line_styles['arima'], 
                      linewidth=2, label='ARIMA Adaptive')
    axes[0, 0].plot(lstm_df.index, lstm_df['waiting_time'].rolling(window).mean(),
                      color=colors['lstm'], linestyle=line_styles['lstm'],
                      linewidth=2, label='LSTM Adaptive')
    axes[0, 0].plot(fixed_df.index, fixed_df['waiting_time'].rolling(window).mean(),
                      color=colors['fixed'], linestyle=line_styles['fixed'],
                      linewidth=2, label='Fixed Timing')
    style_axis(axes[0, 0], 'Average Waiting Time\n(30-step rolling average)',
               'Simulation Step', 'Waiting Time (seconds)')
    
    # Queue Length Plot
    axes[0, 1].plot(arima_df.index, arima_df['queue_length'].rolling(window).mean(), 
                      color=colors['arima'], linestyle=line_styles['arima'],
                      linewidth=2, label='ARIMA Adaptive')
    axes[0, 1].plot(lstm_df.index, lstm_df['queue_length'].rolling(window).mean(),
                      color=colors['lstm'], linestyle=line_styles['lstm'],
                      linewidth=2, label='LSTM Adaptive')
    axes[0, 1].plot(fixed_df.index, fixed_df['queue_length'].rolling(window).mean(),
                      color=colors['fixed'], linestyle=line_styles['fixed'],
                      linewidth=2, label='Fixed Timing')
    style_axis(axes[0, 1], 'Queue Length\n(30-step rolling average)',
               'Simulation Step', 'Number of Vehicles')
    
    # Travel Time Plot
    axes[1, 0].plot(arima_df.index, arima_df['travel_time'].rolling(window).mean(),
                      color=colors['arima'], linestyle=line_styles['arima'],
                      linewidth=2, label='ARIMA Adaptive')
    axes[1, 0].plot(lstm_df.index, lstm_df['travel_time'].rolling(window).mean(),
                      color=colors['lstm'], linestyle=line_styles['lstm'],
                      linewidth=2, label='LSTM Adaptive')
    axes[1, 0].plot(fixed_df.index, fixed_df['travel_time'].rolling(window).mean(),
                      color=colors['fixed'], linestyle=line_styles['fixed'],
                      linewidth=2, label='Fixed Timing')
    style_axis(axes[1, 0], 'Travel Time\n(30-step rolling average)',
               'Simulation Step', 'Travel Time (seconds)')
    
    # Improvement Comparison Plot
    axes[1, 1].remove()
    axes[1, 1] = fig.add_subplot(224)
    
    def safe_improvement(adaptive, fixed):
        with np.errstate(divide='ignore', invalid='ignore'):
            improvement = np.where(fixed > 0, (fixed - adaptive) / fixed * 100, 0)
        return np.nan_to_num(np.clip(improvement, -100, 100))
    
    arima_wait_imp = safe_improvement(arima_df['waiting_time'], fixed_df['waiting_time'])
    lstm_wait_imp = safe_improvement(lstm_df['waiting_time'], fixed_df['waiting_time'])
    arima_queue_imp = safe_improvement(arima_df['queue_length'], fixed_df['queue_length'])
    lstm_queue_imp = safe_improvement(lstm_df['queue_length'], fixed_df['queue_length'])
    arima_travel_imp = safe_improvement(arima_df['travel_time'], fixed_df['travel_time'])
    lstm_travel_imp = safe_improvement(lstm_df['travel_time'], fixed_df['travel_time'])
    
    axes[1, 1].plot(arima_df.index, arima_wait_imp, color=colors['arima'], linestyle='-', label='ARIMA Wait')
    axes[1, 1].plot(lstm_df.index, lstm_wait_imp, color=colors['lstm'], linestyle='-', label='LSTM Wait')
    axes[1, 1].plot(arima_df.index, arima_queue_imp, color=colors['arima'], linestyle='--', label='ARIMA Queue')
    axes[1, 1].plot(lstm_df.index, lstm_queue_imp, color=colors['lstm'], linestyle='--', label='LSTM Queue')
    axes[1, 1].plot(arima_df.index, arima_travel_imp, color=colors['arima'], linestyle=':', label='ARIMA Travel')
    axes[1, 1].plot(lstm_df.index, lstm_travel_imp, color=colors['lstm'], linestyle=':', label='LSTM Travel')
    
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    style_axis(axes[1, 1], 'Percentage Improvement vs Fixed Timing',
               'Simulation Step', 'Improvement (%)')
    axes[1, 1].set_ylim(-20, 100)
    
    plt.tight_layout()
    plt.savefig('Generated Visualizations/Traffic_Control_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Visualizations generated successfully')

def create_prediction_accuracy_plot(predictions, actual_flows, output_filename='Generated Visualizations/prediction_accuracy.png'):
    plt.figure(figsize=(14, 8))
    
    x = range(len(predictions))
    plt.plot(x, predictions, 'b-', label='Predicted Traffic Flow', linewidth=2)
    plt.plot(x, actual_flows, 'r-', label='Actual Traffic Flow', linewidth=2)
    
    errors = np.array(predictions) - np.array(actual_flows)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / np.array(actual_flows))) * 100
    
    plt.title(f'Prediction Accuracy - MAE: {mae:.2f}, MAPE: {mape:.2f}%', fontsize=16)
    plt.xlabel('Prediction Interval', fontsize=12)
    plt.ylabel('Vehicle Count', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Prediction accuracy plot saved as {output_filename}')

if __name__ == '__main__':
    print("This is a visualization module for traffic simulation data.")
    print("Import and use the functions from another script.")