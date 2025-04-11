import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('seaborn-v0_8-darkgrid')

def generate_visualizations(arima, lstm, bilstm, gru, bigru, fixed):
    # Style configuration
    colors = {
        'arima': '#1f77b4',   # Blue
        'lstm': '#ff7f0e',    # Orange
        'bilstm': '#2ca02c',  # Green
        'gru': '#d62728',     # Red
        'bigru': '#9467bd',   # Purple
        'fixed': '#7f7f7f'    # Gray
    }
    
    line_styles = {
        'arima': (0, (1, 1)),
        'lstm': (0, (5, 5)),
        'bilstm': (0, (3, 5, 1, 5)),
        'gru': (0, (5, 1)),
        'bigru': (0, (3, 1, 1, 1)),
        'fixed': (0, ())
    }

    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Traffic Control Model Performance Comparison', y=0.99, fontsize=16)
    
    # Plotting function
    def plot_metric(ax, metric, title, ylabel):
        window = 30
        for model, df in models.items():
            if model == 'fixed': 
                alpha = 0.7
                lw = 1.5
            else:
                alpha = 0.9
                lw = 2
            ax.plot(df['timestamp'], df[metric].rolling(window).mean(),
                    color=colors[model], 
                    linestyle=line_styles[model],
                    lw=lw,
                    alpha=alpha,
                    label=model.upper())
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)

    # Plot metrics
    models = {'arima': arima, 'lstm': lstm, 'bilstm': bilstm, 
             'gru': gru, 'bigru': bigru, 'fixed': fixed}
    
    plot_metric(axs[0,0], 'waiting_time', 'Waiting Time Comparison', 'Seconds')
    plot_metric(axs[0,1], 'queue_length', 'Queue Length Comparison', 'Vehicles')
    plot_metric(axs[1,0], 'travel_time', 'Travel Time Comparison', 'Seconds')
    
    # Improvement plot
    ax = axs[1,1]
    window = 60
    for model in ['arima', 'lstm', 'bilstm', 'gru', 'bigru']:
        improvement = ((fixed['waiting_time'] - models[model]['waiting_time']) / 
                      fixed['waiting_time'] * 100).rolling(window).mean()
        ax.plot(improvement, 
               color=colors[model],
               linestyle=line_styles[model],
               label=f'{model.upper()} Improvement')
    ax.set_title('Waiting Time Improvement vs Fixed Timing', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=10)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Prediction accuracy
    ax = axs[2,0]
    models_to_plot = ['lstm', 'bilstm', 'gru', 'bigru']
    for model in models_to_plot:
        ax.plot(models[model]['predictions'], 
               color=colors[model],
               linestyle=line_styles[model],
               label=f'{model.upper()} Predictions')
    ax.plot(fixed['actual_flows'], 'k--', label='Actual Traffic', alpha=0.7)
    ax.set_title('Traffic Flow Predictions vs Actual', fontsize=12)
    ax.set_ylabel('Vehicle Count', fontsize=10)
    
    # Create unified legend
    legend_elements = [Line2D([0], [0], color=colors[m], linestyle=line_styles[m],
                      lw=2, label=m.upper()) for m in colors]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Generated Visualizations/full_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual prediction plots
    for model in models_to_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(models[model]['predictions'], label='Predictions', color=colors[model])
        plt.plot(models[model]['actual_flows'], label='Actual', color='black', alpha=0.7)
        plt.title(f'{model.upper()} Prediction Accuracy', fontsize=14)
        plt.ylabel('Vehicle Count')
        plt.legend()
        plt.savefig(f'Generated Visualizations/{model}_predictions.png', dpi=150)
        plt.close()

def create_prediction_accuracy_plot(predictions, actuals, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predictions', alpha=0.8)
    plt.plot(actuals, label='Actual', alpha=0.6)
    plt.title('Prediction Accuracy')
    plt.legend()
    plt.savefig(filename)
    plt.close()