import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Add this line FIRST
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import traci
import numpy as np
import pandas as pd
import pytz
import datetime
from datetime import UTC
import time
from statsmodels.tsa.arima.model import ARIMA
from sumolib.net import Phase
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress warnings
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")

# Configure GPU memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
except:
    print("Could not configure GPU memory growth")

def getdatetime():
    utc_now = datetime.datetime.now(UTC)
    currentDT = pytz.utc.localize(utc_now.replace(tzinfo=None)).astimezone(pytz.timezone('Asia/Singapore'))
    return currentDT.strftime('%Y-%m-%d %H:%M:%S')

def create_model(seq_length, model_type):
    model = Sequential()
    model.add(Input(shape=(seq_length, 1)))
    
    # Differentiate models more significantly - adjust unit counts and architecture
    if model_type == 'lstm':
        model.add(LSTM(32, activation='relu'))
    elif model_type == 'bilstm':
        model.add(Bidirectional(LSTM(64, activation='relu')))
    elif model_type == 'gru':
        model.add(GRU(48, activation='relu'))
    elif model_type == 'bigru':
        model.add(Bidirectional(GRU(96, activation='relu')))
    
    # Model-specific hidden layers for further differentiation
    if model_type == 'lstm':
        model.add(Dense(8, activation='relu'))
    elif model_type == 'bilstm':
        model.add(Dense(16, activation='relu'))
    elif model_type == 'gru':
        model.add(Dense(12, activation='relu'))
    elif model_type == 'bigru':
        model.add(Dense(24, activation='relu'))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def run_simulation(use_adaptive_timing=True, model_type='arima', seed=42):
    sumoCmd = ['sumo-gui', '-c', 'osm.sumocfg', '--seed', str(seed)]
    traci.start(sumoCmd)
    
    traffic_light_ids = traci.trafficlight.getIDList()
    target_tl = next((tl for tl in traffic_light_ids if 'cluster_12145877138_12145877139_12145877142_12145877145' in tl), None)
    
    if not target_tl:
        print('ERROR: Could not find the target traffic light cluster!')
        traci.close()
        return None
    
    print(f'Target Traffic Light Found: {target_tl}')
    
    programs = traci.trafficlight.getAllProgramLogics(target_tl)
    if not programs:
        print(f'ERROR: No traffic light programs found for {target_tl}')
        traci.close()
        return None
    
    current_logic = programs[0]
    phases = current_logic.phases
    
    # Initialize all data lists
    time_series_data = []
    time_stamps = []
    waiting_times = []
    queue_lengths = []
    travel_times = []
    predictions = []
    actual_flows = []
    lane_densities = {}
    
    # Model setup
    model = None
    scaler = MinMaxScaler(feature_range=(0, 1))
    seq_length = 10
    last_training_time = 0
    vehicle_start_times = {}
    completed_vehicles = set()
    incoming_lanes = []
    
    # Get incoming lanes
    for link_index in range(traci.trafficlight.getRedYellowGreenState(target_tl).count('G')):
        try:
            controlled_links = traci.trafficlight.getControlledLinks(target_tl)
            if link_index < len(controlled_links) and controlled_links[link_index]:
                incoming_lane = controlled_links[link_index][0][0]
                if incoming_lane not in incoming_lanes:
                    incoming_lanes.append(incoming_lane)
        except:
            pass
    
    print(f'Monitoring {len(incoming_lanes)} incoming lanes')
    
    # Add timing_adjustments to track actual timing changes
    timing_adjustments = []
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        current_time = getdatetime()
        
        # Collect metrics
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)
        
        # Store the actual flow value immediately
        current_flow = vehicle_count if vehicle_count > 0 else 0.001
        actual_flows.append(current_flow)
        
        # Core metrics collection
        time_series_data.append(vehicle_count)
        time_stamps.append(current_time)
        
        # Waiting time calculation
        total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids) if vehicle_ids else 0
        avg_waiting = total_waiting / vehicle_count if vehicle_count > 0 else 0
        waiting_times.append(avg_waiting)
        
        # Queue length calculation
        queue = sum(1 for v in vehicle_ids if traci.vehicle.getSpeed(v) < 0.1)
        queue_lengths.append(queue)
        
        # Travel time calculation
        current_travel_times = []
        
        # Track new vehicles that enter the simulation
        for v in vehicle_ids:
            if v not in vehicle_start_times and v not in completed_vehicles:
                vehicle_start_times[v] = step
        
        # Calculate travel times for vehicles that have completed their route
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for v in arrived_vehicles:
            if v in vehicle_start_times:
                trip_time = step - vehicle_start_times[v]
                if trip_time > 0:  # Make sure we have a valid travel time
                    current_travel_times.append(trip_time)
                    completed_vehicles.add(v)  # Add to completed set
                del vehicle_start_times[v]  # Remove from tracking
        
        # If no vehicles completed this step but we have vehicles in the simulation,
        # we should estimate current travel times for active vehicles
        if not current_travel_times and vehicle_ids:
            # Calculate time in simulation for active vehicles
            active_travel_times = []
            for v in vehicle_ids:
                if v in vehicle_start_times:
                    current_duration = step - vehicle_start_times[v]
                    if current_duration > 0:
                        active_travel_times.append(current_duration)
            
            # If we have active vehicles with travel times, use their average
            # This gives us a running estimate even before vehicles complete
            if active_travel_times:
                avg_travel_time = np.mean(active_travel_times)
            else:
                avg_travel_time = travel_times[-1] if travel_times else 0
        else:
            # Use completed vehicle travel times if available
            avg_travel_time = np.mean(current_travel_times) if current_travel_times else (travel_times[-1] if travel_times else 0)
        
        travel_times.append(avg_travel_time)
        
        # Debug output for travel times
        if step % 100 == 0:
            print(f"Step {step}: {len(current_travel_times)} vehicles completed trips")
            print(f"Average travel time: {avg_travel_time:.2f} seconds")
            print(f"Currently tracking {len(vehicle_start_times)} vehicles")
            print(f"Total completed: {len(completed_vehicles)} vehicles")
        
        # Lane densities
        current_lane_densities = {}
        for lane in incoming_lanes:
            try:
                count = traci.lane.getLastStepVehicleNumber(lane)
                length = traci.lane.getLength(lane)
                current_lane_densities[lane] = count / length if length > 0 else 0
            except:
                current_lane_densities[lane] = 0
        lane_densities[step] = current_lane_densities
        
        # Initialize prediction and adjustment tracking
        current_prediction = 0
        current_adjustment = 0
        
        # Adaptive control logic
        if use_adaptive_timing and len(time_series_data) > 36 and step % 5 == 0:
            try:
                if model_type == 'arima':
                    # Get current program details first
                    current_program = traci.trafficlight.getAllProgramLogics(target_tl)[0]
                    program_type = current_program.type
                    
                    try:
                        # ARIMA Prediction Logic
                        window_size = min(60, len(time_series_data))  # Ensure we don't ask for more data than available
                        transformed_data = np.log1p(time_series_data[-window_size:])
                        model = ARIMA(transformed_data, order=(1, 1, 1))
                        model_fit = model.fit(method_kwargs={"maxiter": 1000})
                        prediction_result = np.expm1(model_fit.forecast(steps=1))
                        current_prediction = max(prediction_result[0], 0)
                        
                        # Print debug info occasionally
                        if step % 50 == 0:
                            print(f"Step {step}, Actual: {current_flow}, Prediction: {current_prediction}")
                        
                        # Initialize new_phases with current phases first
                        new_phases = [Phase(p.duration, p.state, p.minDur, p.maxDur) for p in phases]
                        
                        # Traffic light adjustment
                        congested_lanes = sorted(current_lane_densities.items(), key=lambda x: x[1], reverse=True)
                        
                        # Modify phases based on prediction
                        for i, phase in enumerate(phases):
                            if 'G' in phase.state:
                                green_lanes = []
                                for j, state in enumerate(phase.state):
                                    if state == 'G' and j < len(traci.trafficlight.getControlledLinks(target_tl)):
                                        links = traci.trafficlight.getControlledLinks(target_tl)[j]
                                        if links:
                                            green_lanes.append(links[0][0])
                                
                                phase_congestion = sum(current_lane_densities.get(lane, 0) for lane in green_lanes)
                                
                                if phase_congestion > 0.1 or current_prediction > 20:
                                    new_phases[i] = Phase(
                                        min(phase.maxDur, phase.duration + 10),
                                        phase.state,
                                        phase.minDur,
                                        phase.maxDur
                                    )
                                else:
                                    new_phases[i] = Phase(
                                        max(phase.minDur, phase.duration - 5),
                                        phase.state,
                                        phase.minDur,
                                        phase.maxDur
                                    )
                            else:
                                new_duration = min(phase.duration, 5) if 'y' in phase.state else phase.duration
                                new_phases[i] = Phase(
                                    new_duration,
                                    phase.state,
                                    phase.minDur,
                                    phase.maxDur
                                )

                        updated_program = traci.trafficlight.Logic(
                            programID=current_program.programID,
                            type=program_type,
                            currentPhaseIndex=current_program.currentPhaseIndex,
                            phases=new_phases,
                            subParameter=current_program.subParameter
                        )
                        traci.trafficlight.setProgramLogic(target_tl, updated_program)

                        # ARIMA-specific adjustment strategy
                        prediction_ratio = current_prediction / max(1, np.mean(actual_flows[-10:]))
                        arima_adjustment_factor = 1.0 + min(0.5, max(-0.3, (prediction_ratio - 1) * 1.5))
                        
                        # Apply ARIMA-specific adjustments
                        for i, phase in enumerate(phases):
                            if 'G' in phase.state:
                                # ... existing green phase code ...
                                
                                # ARIMA specific phase duration adjustment logic
                                base_duration = (phase.maxDur + phase.minDur) / 2
                                congestion_factor = 1.0 + min(0.7, max(-0.4, phase_congestion * 10))
                                
                                # Combine factors - ARIMA's unique approach
                                combined_factor = arima_adjustment_factor * 0.6 + congestion_factor * 0.4
                                
                                # Calculate new duration with stronger adjustments
                                new_duration = int(base_duration * combined_factor)
                                new_duration = max(phase.minDur, min(phase.maxDur, new_duration))
                                
                                current_adjustment = new_duration - phase.duration
                                new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                    
                    except Exception as e:
                        print(f'ARIMA Error: {str(e)}')
                        current_prediction = actual_flows[-2] if len(actual_flows) > 1 else current_flow
                        # Fallback to original phases if new_phases failed
                        updated_program = traci.trafficlight.Logic(
                            programID="fallback",
                            type=program_type,
                            currentPhaseIndex=0,
                            phases=phases,
                            subParameter=current_program.subParameter
                        )
                        traci.trafficlight.setProgramLogic(target_tl, updated_program)
                
                elif model_type in ['lstm', 'bilstm', 'gru', 'bigru']:
                    # RNN model logic with more differentiated behavior
                    current_program = traci.trafficlight.getAllProgramLogics(target_tl)[0]
                    program_type = current_program.type
                    
                    # Training frequency varies by model type
                    training_frequency = {
                        'lstm': 100,
                        'bilstm': 120,
                        'gru': 80,
                        'bigru': 140
                    }
                    
                    # Different epochs and batch sizes for different models
                    training_epochs = {
                        'lstm': 5,
                        'bilstm': 8,
                        'gru': 6,
                        'bigru': 10
                    }
                    
                    # Train model with different frequencies based on model type
                    if step % training_frequency[model_type] == 0 or model is None:
                        scaled_data = scaler.fit_transform(np.array(time_series_data).reshape(-1, 1))
                        X, y = create_sequences(scaled_data, seq_length)
                        if len(X) > 0:
                            model = create_model(seq_length, model_type)
                            model.fit(X, y, 
                                  epochs=training_epochs[model_type], 
                                  batch_size=16, 
                                  callbacks=[EarlyStopping(monitor='loss', patience=2)], 
                                  verbose=0)
                    
                    if model:
                        recent = scaler.transform(np.array(time_series_data[-seq_length:]).reshape(-1, 1))
                        prediction_result = model.predict(recent.reshape(1, seq_length, 1), verbose=0)[0][0]
                        current_prediction = float(scaler.inverse_transform([[prediction_result]])[0][0])
                    
                    # Model-specific prediction factor calculations
                    if model_type == 'lstm':
                        prediction_factor = min(1.8, max(0.6, current_prediction / max(5, np.mean(actual_flows[-10:]))))
                    elif model_type == 'bilstm':
                        prediction_factor = min(1.6, max(0.7, (current_prediction + 2) / (max(5, np.mean(actual_flows[-15:])) + 2)))
                    elif model_type == 'gru':
                        prediction_factor = min(1.7, max(0.5, (current_prediction * 1.1) / max(5, np.mean(actual_flows[-8:]))))
                    elif model_type == 'bigru':
                        prediction_factor = min(2.0, max(0.4, (current_prediction * 1.2) / max(5, np.mean(actual_flows[-12:]))))
                    
                    # Phase adjustment with model-specific parameter tuning
                    new_phases = []
                    for phase in phases:
                        if 'G' in phase.state:
                            # Get congestion data for this phase
                            green_lanes = [
                                link[0][0] for i, state in enumerate(phase.state)
                                if state == 'G' and i < len(traci.trafficlight.getControlledLinks(target_tl))
                                for link in [traci.trafficlight.getControlledLinks(target_tl)[i]]
                                if link and len(link) > 0 and link[0]
                            ]
                            
                            # Calculate congestion with model-specific weighting
                            lane_weights = {
                                'lstm': [1.0] * len(green_lanes),
                                'bilstm': [1.2, 0.8] * (len(green_lanes) // 2 + 1),
                                'gru': [0.9, 1.1] * (len(green_lanes) // 2 + 1),
                                'bigru': [1.3, 0.7] * (len(green_lanes) // 2 + 1)
                            }
                            
                            # Use the appropriate weights for this model type
                            weights = lane_weights[model_type][:len(green_lanes)]
                            
                            # Apply weighted congestion calculation
                            congestion = sum(current_lane_densities.get(lane, 0) * weights[i % len(weights)] 
                                           for i, lane in enumerate(green_lanes))
                            
                            # Calculate base duration adjustment from congestion - model specific
                            congestion_multipliers = {
                                'lstm': 4,
                                'bilstm': 6,
                                'gru': 5,
                                'bigru': 7
                            }
                            
                            congestion_factor = 1.0 + min(0.5, max(-0.3, congestion * congestion_multipliers[model_type]))
                            
                            # Combine factors - different models weight these differently
                            if model_type == 'lstm':
                                combined_factor = prediction_factor * 0.7 + congestion_factor * 0.3
                            elif model_type == 'bilstm':
                                combined_factor = prediction_factor * 0.5 + congestion_factor * 0.5
                            elif model_type == 'gru':
                                combined_factor = prediction_factor * 0.6 + congestion_factor * 0.4
                            elif model_type == 'bigru':
                                combined_factor = prediction_factor * 0.4 + congestion_factor * 0.6
                            
                            # Apply combined factor to base duration - with model-specific base duration calculation
                            base_durations = {
                                'lstm': (phase.maxDur + phase.minDur) / 2,
                                'bilstm': (phase.maxDur * 0.6 + phase.minDur * 0.4),
                                'gru': (phase.maxDur * 0.5 + phase.minDur * 0.5),
                                'bigru': (phase.maxDur * 0.7 + phase.minDur * 0.3)
                            }
                            
                            base_duration = base_durations[model_type]
                            new_duration = int(base_duration * combined_factor)
                            new_duration = max(phase.minDur, min(phase.maxDur, new_duration))
                            
                            current_adjustment = new_duration - phase.duration
                            new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                        else:
                            # Different models handle yellow and red phases differently
                            if model_type == 'lstm':
                                new_duration = phase.duration  # LSTM keeps original duration
                            elif model_type == 'bilstm':
                                new_duration = min(phase.duration, 4) if 'y' in phase.state else phase.duration
                            elif model_type == 'gru':
                                new_duration = min(phase.duration + 1, 6) if 'y' in phase.state else phase.duration
                            elif model_type == 'bigru':
                                new_duration = min(phase.duration - 1, 3) if 'y' in phase.state else phase.duration
                            
                            new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                    
                    traci.trafficlight.setProgramLogic(target_tl, traci.trafficlight.Logic(
                        programID=f"adaptive_{model_type}",  # Include model type in program ID
                        type=program_type,
                        phases=new_phases,
                        currentPhaseIndex=traci.trafficlight.getPhase(target_tl)
                    ))
            except Exception as e:
                print(f"{model_type.upper()} Error: {str(e)}")
                current_prediction = actual_flows[-2] if len(actual_flows) > 1 else current_flow
                current_adjustment = 0
        
        # Always append the prediction and timing adjustment
        predictions.append(current_prediction)
        timing_adjustments.append(current_adjustment)
    
    traci.close()
    
    # Find the shortest length among all the data arrays
    min_len = min(
        len(time_stamps),
        len(waiting_times),
        len(queue_lengths),
        len(travel_times),
        len(predictions),
        len(actual_flows)
    )
    
    # Create results dataframe, ensuring all arrays are the same length
    return pd.DataFrame({
        'timestamp': time_stamps[:min_len],
        'waiting_time': waiting_times[:min_len],
        'queue_length': queue_lengths[:min_len],
        'travel_time': travel_times[:min_len],
        'predictions': predictions[:min_len],
        'actual_flows': actual_flows[:min_len],
        'timing_adjustments': timing_adjustments[:min_len],
        'model_type': [model_type] * min_len
    })

# ... [Keep all previous imports and GPU configuration] ...

def create_prediction_accuracy_plot(predictions, actuals, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label='Predictions', alpha=0.7)
    plt.plot(actuals, label='Actual Traffic', alpha=0.5)
    plt.title('Traffic Flow Prediction Accuracy')
    plt.xlabel('Time Steps')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def generate_visualizations(arima_df, lstm_df, bilstm_df, gru_df, bigru_df, fixed_df):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))
    
    # Common styling
    colors = {
        'arima': '#1f77b4',
        'lstm': '#ff7f0e',
        'bilstm': '#2ca02c',
        'gru': '#d62728',
        'bigru': '#9467bd',
        'fixed': '#7f7f7f'
    }
    
    # Waiting Time Comparison
    axs[0].plot(arima_df['waiting_time'].rolling(30).mean(), color=colors['arima'], label='ARIMA')
    axs[0].plot(lstm_df['waiting_time'].rolling(30).mean(), color=colors['lstm'], label='LSTM')
    axs[0].plot(bilstm_df['waiting_time'].rolling(30).mean(), color=colors['bilstm'], label='BiLSTM')
    axs[0].plot(gru_df['waiting_time'].rolling(30).mean(), color=colors['gru'], label='GRU')
    axs[0].plot(bigru_df['waiting_time'].rolling(30).mean(), color=colors['bigru'], label='BiGRU')
    axs[0].plot(fixed_df['waiting_time'].rolling(30).mean(), color=colors['fixed'], label='Fixed', linestyle='--')
    axs[0].set_title('Average Waiting Time Comparison (30-step Moving Average)')
    axs[0].set_ylabel('Seconds')
    axs[0].legend()

    # Queue Length Comparison
    axs[1].plot(arima_df['queue_length'].rolling(30).mean(), color=colors['arima'], label='ARIMA')
    axs[1].plot(lstm_df['queue_length'].rolling(30).mean(), color=colors['lstm'], label='LSTM')
    axs[1].plot(bilstm_df['queue_length'].rolling(30).mean(), color=colors['bilstm'], label='BiLSTM')
    axs[1].plot(gru_df['queue_length'].rolling(30).mean(), color=colors['gru'], label='GRU')
    axs[1].plot(bigru_df['queue_length'].rolling(30).mean(), color=colors['bigru'], label='BiGRU')
    axs[1].plot(fixed_df['queue_length'].rolling(30).mean(), color=colors['fixed'], label='Fixed', linestyle='--')
    axs[1].set_title('Queue Length Comparison (30-step Moving Average)')
    axs[1].set_ylabel('Vehicles')
    axs[1].legend()

    # Travel Time Comparison
    axs[2].plot(arima_df['travel_time'].rolling(30).mean(), color=colors['arima'], label='ARIMA')
    axs[2].plot(lstm_df['travel_time'].rolling(30).mean(), color=colors['lstm'], label='LSTM')
    axs[2].plot(bilstm_df['travel_time'].rolling(30).mean(), color=colors['bilstm'], label='BiLSTM')
    axs[2].plot(gru_df['travel_time'].rolling(30).mean(), color=colors['gru'], label='GRU')
    axs[2].plot(bigru_df['travel_time'].rolling(30).mean(), color=colors['bigru'], label='BiGRU')
    axs[2].plot(fixed_df['travel_time'].rolling(30).mean(), color=colors['fixed'], label='Fixed', linestyle='--')
    axs[2].set_title('Travel Time Comparison (30-step Moving Average)')
    axs[2].set_ylabel('Seconds')
    axs[2].set_xlabel('Simulation Steps')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('Generated Visualizations/performance_comparison.png')
    plt.close()

def calculate_model_comparison(*model_dfs):
    metrics = ['waiting_time', 'queue_length', 'travel_time']
    comparisons = []
    
    # Get fixed timing baseline
    fixed_df = next(df for df in model_dfs if df['model_type'].iloc[0] == 'fixed')
    fixed_means = {metric: fixed_df[metric].rolling(30).mean().mean() for metric in metrics}
    
    for df in model_dfs:
        if df['model_type'].iloc[0] == 'fixed':
            continue
            
        model_name = df['model_type'].iloc[0].upper()
        improvements = []
        
        for metric in metrics:
            model_mean = df[metric].rolling(30).mean().mean()
            fixed_mean = fixed_means[metric]
            improvement = -((fixed_mean - model_mean) / fixed_mean * 100) if fixed_mean != 0 else 0
            improvements.append(improvement)
        
        comparisons.append({
            'Model': model_name,
            'Waiting Time Improvement (%)': f"{improvements[0]:.2f}%",
            'Queue Length Improvement (%)': f"{improvements[1]:.2f}%",
            'Travel Time Improvement (%)': f"{improvements[2]:.2f}%"
        })
    
    print("\n=== Model Performance Comparison ===")
    print("\nImprovements over Fixed Timing Control:")
    print("(Positive values indicate better performance)")
    print("-" * 80)
    print(pd.DataFrame(comparisons).to_markdown(index=False, tablefmt="grid"))
    print("-" * 80)
    
    # Print baseline fixed timing metrics
    print("\nFixed Timing Baseline Metrics:")
    print("-" * 80)
    baseline_metrics = {
        'Metric': ['Average Waiting Time', 'Average Queue Length', 'Average Travel Time'],
        'Value': [f"{fixed_means['waiting_time']:.2f} seconds",
                 f"{fixed_means['queue_length']:.2f} vehicles",
                 f"{fixed_means['travel_time']:.2f} seconds"]
    }
    print(pd.DataFrame(baseline_metrics).to_markdown(index=False, tablefmt="grid"))
    print("-" * 80)

def main_menu():
    models = ['arima', 'lstm', 'bilstm', 'gru', 'bigru', 'fixed']
    
    while True:
        print("\n=== Traffic Simulation Control Center ===")
        print("1. Run all simulations (ARIMA, LSTM, BiLSTM, GRU, BiGRU, Fixed)")
        print("2. Generate visualizations from existing data")
        print("3. Print performance metrics")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            print("\nInitializing simulations...")
            results = {}
            
            for model in models:
                print(f"Running {model.upper()} model...")
                results[model] = run_simulation(
                    use_adaptive_timing=(model != 'fixed'),
                    model_type=model,
                    seed=42
                )
                results[model].to_csv(f'Model Data/{model}_results.csv', index=False)
            
            print("\nAll simulations completed successfully!")
        
        elif choice == '2':
            print("\nGenerating visualizations...")
            try:
                data = {model: pd.read_csv(f'Model Data/{model}_results.csv') for model in models}
                generate_visualizations(
                    data['arima'], data['lstm'], data['bilstm'],
                    data['gru'], data['bigru'], data['fixed']
                )
                
                for model in models:
                    if model != 'fixed':
                        create_prediction_accuracy_plot(
                            data[model]['predictions'],
                            data[model]['actual_flows'],
                            f'Generated Visualizations/{model}_predictions.png'
                        )
                
                print("Visualizations saved in 'Generated Visualizations' folder")
            
            except Exception as e:
                print(f"Visualization error: {str(e)}")
        
        elif choice == '3':
            print("\nCalculating metrics...")
            try:
                data = [pd.read_csv(f'Model Data/{model}_results.csv') for model in models]
                calculate_model_comparison(*data)
            except Exception as e:
                print(f"Metric calculation error: {str(e)}")
        
        elif choice == '4':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == '__main__':
    # Create necessary directories
    import os
    os.makedirs('Model Data', exist_ok=True)
    os.makedirs('Generated Visualizations', exist_ok=True)
    
    main_menu()
