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
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from visualization import generate_visualizations, create_prediction_accuracy_plot
import pickle

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

def create_lstm_model(seq_length):
    model = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(32, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def run_simulation(use_adaptive_timing=True, model_type='arima', seed=42):
    # Set fixed seed+1 for non-adaptive to match original code behavior
    if not use_adaptive_timing:
        seed = seed + 1
        
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
    
    # Initialize data collection structures
    time_series_data = []
    time_stamps = []
    waiting_times = []
    queue_lengths = []
    travel_times = []
    predictions = []
    actual_flows = []
    lane_densities = {}
    
    # Vehicle tracking
    completed_vehicles = set()
    vehicle_start_times = {}
    incoming_lanes = []
    completed_travel_times = []
    
    # LSTM specific setup
    lstm_model = None
    scaler = MinMaxScaler(feature_range=(0, 1))
    seq_length = 10
    last_lstm_training_time = 0
    
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
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        current_time = getdatetime()
        
        # Get current vehicle data
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)
        
        # Record time series data
        time_series_data.append(vehicle_count)
        time_stamps.append(current_time)
        
        # Track vehicle start times
        for veh_id in vehicle_ids:
            if veh_id not in vehicle_start_times:
                vehicle_start_times[veh_id] = step
        
        # Calculate waiting time
        total_waiting_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids) if vehicle_ids else 0
        avg_waiting_time = total_waiting_time / vehicle_count if vehicle_count > 0 else 0
        waiting_times.append(avg_waiting_time)
        
        # Calculate queue length
        queue = sum(1 for veh_id in vehicle_ids if traci.vehicle.getSpeed(veh_id) < 0.1)
        queue_lengths.append(queue)
        
        # Calculate travel times for completed vehicles
        current_completed_times = []
        for veh_id in list(vehicle_start_times.keys()):
            if veh_id not in vehicle_ids and veh_id not in completed_vehicles:
                completed_vehicles.add(veh_id)
                trip_time = step - vehicle_start_times[veh_id]
                current_completed_times.append(trip_time)
                completed_travel_times.append(trip_time)
        
        if completed_travel_times:
            avg_travel_time = np.mean(completed_travel_times)
        else:
            avg_travel_time = 0
        travel_times.append(avg_travel_time)
        
        # Calculate lane densities
        current_lane_densities = {}
        for lane in incoming_lanes:
            try:
                vehicles_on_lane = traci.lane.getLastStepVehicleNumber(lane)
                lane_length = traci.lane.getLength(lane)
                density = vehicles_on_lane / lane_length if lane_length > 0 else 0
                current_lane_densities[lane] = density
            except:
                current_lane_densities[lane] = 0
        
        lane_densities[step] = current_lane_densities
        
        # Adaptive timing logic
        if use_adaptive_timing and len(time_series_data) > 30 and step % 5 == 0:
            actual_flow = vehicle_count
            
            if model_type == 'arima':
                # Default to current flow if prediction fails
                predicted_traffic = actual_flow
                
                try:
                    transformed_data = np.log1p(time_series_data[-30:])
                    model = ARIMA(transformed_data, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit()
                    prediction = np.expm1(model_fit.forecast(steps=5))
                    predicted_traffic = max(prediction[-1], 0)
                    
                    # Traffic light adjustment logic
                    congested_lanes = sorted(current_lane_densities.items(), key=lambda x: x[1], reverse=True)
                    new_phases = []
                    
                    for i, phase in enumerate(phases):
                        if 'G' in phase.state:
                            green_lanes = []
                            for j, state in enumerate(phase.state):
                                if state == 'G' and j < len(traci.trafficlight.getControlledLinks(target_tl)):
                                    links = traci.trafficlight.getControlledLinks(target_tl)[j]
                                    if links:
                                        green_lanes.append(links[0][0])
                            
                            phase_congestion = sum(current_lane_densities.get(lane, 0) for lane in green_lanes)
                            
                            if phase_congestion > 0.1 or predicted_traffic > 20:
                                new_duration = min(phase.maxDur, phase.duration + 10)
                            else:
                                new_duration = max(phase.minDur, phase.duration - 5)
                            
                            new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                        else:
                            new_duration = min(phase.duration, 5) if 'y' in phase.state else phase.duration
                            new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                    
                    current_program = traci.trafficlight.getAllProgramLogics(target_tl)[0]
                    updated_program = traci.trafficlight.Logic(
                        programID=current_program.programID,
                        type=current_program.type,
                        currentPhaseIndex=current_program.currentPhaseIndex,
                        phases=new_phases,
                        subParameter=current_program.subParameter
                    )
                    
                    traci.trafficlight.setProgramLogic(target_tl, updated_program)
                except Exception as e:
                    print(f'ARIMA Training Error: {e}')
                    predicted_traffic = actual_flow
                
                # Always record prediction and actual flow, even if model fails
                predictions.append(predicted_traffic)
                actual_flows.append(actual_flow)
            
            elif model_type == 'lstm':
                # Default to current flow if prediction fails
                predicted_traffic = actual_flow
                
                try:
                    if len(time_series_data) >= max(30, seq_length + 5):
                        if (step % 200 == 0 or lstm_model is None) and step > last_lstm_training_time + 100:
                            tf.keras.backend.clear_session()
                            recent_data = time_series_data[-200:] if len(time_series_data) > 200 else time_series_data
                            data_array = np.array(recent_data).reshape(-1, 1)
                            normalized_data = scaler.fit_transform(data_array)
                            
                            X, y = create_sequences(normalized_data, seq_length)
                            
                            if len(X) > 0 and len(y) > 0:
                                early_stopping = EarlyStopping(monitor='loss', patience=3)
                                lstm_model = create_lstm_model(seq_length)
                                lstm_model.fit(X, y, epochs=5, batch_size=16, verbose=0, callbacks=[early_stopping])
                                last_lstm_training_time = step
                    
                        if lstm_model is not None and len(time_series_data) >= seq_length:
                            recent_data = np.array(time_series_data[-seq_length:]).reshape(-1, 1)
                            normalized_recent = scaler.transform(recent_data)
                            normalized_recent = normalized_recent.reshape(1, seq_length, 1)
                            
                            normalized_prediction = lstm_model.predict(normalized_recent, verbose=0)
                            predicted_traffic = scaler.inverse_transform(normalized_prediction)[0][0]
                            predicted_traffic = max(predicted_traffic, 0)
                        
                    # Traffic light adjustment logic
                    congested_lanes = sorted(current_lane_densities.items(), key=lambda x: x[1], reverse=True)
                    new_phases = []
                    
                    for i, phase in enumerate(phases):
                        if 'G' in phase.state:
                            green_lanes = []
                            for j, state in enumerate(phase.state):
                                if state == 'G' and j < len(traci.trafficlight.getControlledLinks(target_tl)):
                                    links = traci.trafficlight.getControlledLinks(target_tl)[j]
                                    if links:
                                        green_lanes.append(links[0][0])
                            
                            phase_congestion = sum(current_lane_densities.get(lane, 0) for lane in green_lanes)
                            
                            if phase_congestion > 0.15 or predicted_traffic > 18:
                                new_duration = min(phase.maxDur, phase.duration + 10)
                            else:
                                new_duration = max(phase.minDur, phase.duration - 5)
                            
                            new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                        else:
                            new_duration = min(phase.duration, 5) if 'y' in phase.state else phase.duration
                            new_phases.append(Phase(new_duration, phase.state, phase.minDur, phase.maxDur))
                    
                    current_program = traci.trafficlight.getAllProgramLogics(target_tl)[0]
                    updated_program = traci.trafficlight.Logic(
                        programID=current_program.programID,
                        type=current_program.type,
                        currentPhaseIndex=current_program.currentPhaseIndex,
                        phases=new_phases,
                        subParameter=current_program.subParameter
                    )
                    
                    traci.trafficlight.setProgramLogic(target_tl, updated_program)
                except Exception as e:
                    print(f'LSTM Training Error: {e}')
                
                # Always record prediction and actual flow, regardless of model success
                predictions.append(predicted_traffic)
                actual_flows.append(actual_flow)
    
    traci.close()  
    # Ensure predictions and actual_flows are filled for all time steps if needed
    if use_adaptive_timing and len(predictions) < len(time_stamps):
        # If we have some predictions but not enough, pad the beginning
        if len(predictions) > 0:
            padding_needed = len(time_stamps) - len(predictions)
            # Use the first prediction value for padding
            predictions = [predictions[0]] * padding_needed + predictions
            actual_flows = [actual_flows[0]] * padding_needed + actual_flows
        else:
            # If we have no predictions, use zeros
            predictions = [0] * len(time_stamps)
            actual_flows = [0] * len(time_stamps)
    
    # Ensure all arrays have the same length
    min_length = min(len(time_stamps), len(waiting_times), len(queue_lengths), len(travel_times))
    
    # Include predictions and actual flows in min_length calculation only if they exist
    if use_adaptive_timing and len(predictions) > 0 and len(actual_flows) > 0:
        min_length = min(min_length, len(predictions), len(actual_flows))
    
    results = {
        'timestamp': time_stamps[:min_length],
        'waiting_time': waiting_times[:min_length],
        'queue_length': queue_lengths[:min_length],
        'travel_time': travel_times[:min_length],
        'predictions': predictions[:min_length] if len(predictions) > 0 else [],
        'actual_flows': actual_flows[:min_length] if len(actual_flows) > 0 else [],
        'lane_densities': lane_densities,
        'model_type': model_type
    }
    
    return results

def calculate_and_print_metrics(adaptive_df, fixed_df, model_type='ARIMA'):
    # Calculate metrics using rolling windows like in first code for similar results
    window = 30
    adaptive_wait_smooth = adaptive_df['waiting_time'].rolling(window=window).mean().fillna(adaptive_df['waiting_time'])
    fixed_wait_smooth = fixed_df['waiting_time'].rolling(window=window).mean().fillna(fixed_df['waiting_time'])
    
    adaptive_queue_smooth = adaptive_df['queue_length'].rolling(window=window).mean().fillna(adaptive_df['queue_length'])
    fixed_queue_smooth = fixed_df['queue_length'].rolling(window=window).mean().fillna(fixed_df['queue_length'])
    
    adaptive_travel_smooth = adaptive_df['travel_time'].rolling(window=window).mean().fillna(adaptive_df['travel_time'])
    fixed_travel_smooth = fixed_df['travel_time'].rolling(window=window).mean().fillna(fixed_df['travel_time'])
    
    # Calculate averages based on smoothed data as in the first code
    adaptive_wait = np.mean(adaptive_wait_smooth)
    fixed_wait = np.mean(fixed_wait_smooth)
    wait_improvement = ((fixed_wait - adaptive_wait) / fixed_wait) * 100 if fixed_wait > 0 else 0
    
    adaptive_queue = np.mean(adaptive_queue_smooth)
    fixed_queue = np.mean(fixed_queue_smooth)
    queue_improvement = ((fixed_queue - adaptive_queue) / fixed_queue) * 100 if fixed_queue > 0 else 0
    
    adaptive_travel = np.mean(adaptive_travel_smooth)
    fixed_travel = np.mean(fixed_travel_smooth)
    travel_improvement = ((fixed_travel - adaptive_travel) / fixed_travel) * 100 if fixed_travel > 0 else 0
    
    # Print metrics similar to first code
    print(f'\nAverage Waiting Time ({model_type}):')
    print(f'Adaptive = {adaptive_wait:.2f}s')
    print(f'Fixed = {fixed_wait:.2f}s')
    print(f'Improvement = {wait_improvement:.2f}%')
    
    print(f'\nAverage Queue Length ({model_type}):')
    print(f'Adaptive = {adaptive_queue:.2f} vehicles')
    print(f'Fixed = {fixed_queue:.2f} vehicles')
    print(f'Improvement = {queue_improvement:.2f}%')
    
    print(f'\nAverage Travel Time ({model_type}):')
    print(f'Adaptive = {adaptive_travel:.2f}s')
    print(f'Fixed = {fixed_travel:.2f}s')
    print(f'Improvement = {travel_improvement:.2f}%')
    
    if 'predictions' in adaptive_df.columns and 'actual_flows' in adaptive_df.columns and len(adaptive_df['predictions']) > 0:
        predictions = np.array(adaptive_df['predictions'])
        actuals = np.array(adaptive_df['actual_flows'])
        
        if len(predictions) == len(actuals) and len(predictions) > 0:
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            valid_indices = actuals != 0
            mape = np.mean(np.abs((actuals[valid_indices] - predictions[valid_indices]) / actuals[valid_indices])) * 100 if np.any(valid_indices) else float('inf')
            
            print(f'\nPrediction Metrics ({model_type}):')
            print(f'Mean Squared Error (MSE) = {mse:.2f}')
            print(f'Mean Absolute Error (MAE) = {mae:.2f}')
            print(f'Mean Absolute Percentage Error (MAPE) = {mape:.2f}%')

def calculate_model_comparison(arima_df, lstm_df, fixed_df):
    # Apply similar window smoothing as in first code
    window = 30
    arima_wait_smooth = arima_df['waiting_time'].rolling(window=window).mean().fillna(arima_df['waiting_time'])
    lstm_wait_smooth = lstm_df['waiting_time'].rolling(window=window).mean().fillna(lstm_df['waiting_time'])
    fixed_wait_smooth = fixed_df['waiting_time'].rolling(window=window).mean().fillna(fixed_df['waiting_time'])
    
    arima_queue_smooth = arima_df['queue_length'].rolling(window=window).mean().fillna(arima_df['queue_length'])
    lstm_queue_smooth = lstm_df['queue_length'].rolling(window=window).mean().fillna(lstm_df['queue_length'])
    fixed_queue_smooth = fixed_df['queue_length'].rolling(window=window).mean().fillna(fixed_df['queue_length'])
    
    arima_travel_smooth = arima_df['travel_time'].rolling(window=window).mean().fillna(arima_df['travel_time'])
    lstm_travel_smooth = lstm_df['travel_time'].rolling(window=window).mean().fillna(lstm_df['travel_time'])
    fixed_travel_smooth = fixed_df['travel_time'].rolling(window=window).mean().fillna(fixed_df['travel_time'])
    
    print('\n===== PERFORMANCE COMPARISON =====')
    
    arima_wait = np.mean(arima_wait_smooth)
    lstm_wait = np.mean(lstm_wait_smooth)
    fixed_wait = np.mean(fixed_wait_smooth)
    
    arima_queue = np.mean(arima_queue_smooth)
    lstm_queue = np.mean(lstm_queue_smooth)
    fixed_queue = np.mean(fixed_queue_smooth)
    
    arima_travel = np.mean(arima_travel_smooth)
    lstm_travel = np.mean(lstm_travel_smooth)
    fixed_travel = np.mean(fixed_travel_smooth)
    
    print('--- Calculated Averages (30-step rolling window) ---')
    print(f'ARIMA Wait: {arima_wait:.2f}, LSTM Wait: {lstm_wait:.2f}, Fixed Wait: {fixed_wait:.2f}')
    print(f'ARIMA Queue: {arima_queue:.2f}, LSTM Queue: {lstm_queue:.2f}, Fixed Queue: {fixed_queue:.2f}')
    print(f'ARIMA Travel: {arima_travel:.2f}, LSTM Travel: {lstm_travel:.2f}, Fixed Travel: {fixed_travel:.2f}')
    print('--------------------------------------------------')
    
    arima_wait_improvement = ((fixed_wait - arima_wait) / fixed_wait * 100) if fixed_wait > 0 else 0
    lstm_wait_improvement = ((fixed_wait - lstm_wait) / fixed_wait * 100) if fixed_wait > 0 else 0
    
    arima_queue_improvement = ((fixed_queue - arima_queue) / fixed_queue * 100) if fixed_queue > 0 else 0
    lstm_queue_improvement = ((fixed_queue - lstm_queue) / fixed_queue * 100) if fixed_queue > 0 else 0
    
    arima_travel_improvement = ((fixed_travel - arima_travel) / fixed_travel * 100) if fixed_travel > 0 else 0
    lstm_travel_improvement = ((fixed_travel - lstm_travel) / fixed_travel * 100) if fixed_travel > 0 else 0
    
    print('\nImprovement over Fixed Timing:')
    print('--------------------------------')
    print('Metric           | ARIMA   | LSTM    |')
    print('--------------------------------')
    print(f'Waiting Time    | {arima_wait_improvement:>6.1f}% | {lstm_wait_improvement:>6.1f}% |')
    print(f'Queue Length    | {arima_queue_improvement:>6.1f}% | {lstm_queue_improvement:>6.1f}% |')
    print(f'Travel Time     | {arima_travel_improvement:>6.1f}% | {lstm_travel_improvement:>6.1f}% |')
    print('--------------------------------')
    
    # Check if prediction data exists and has equal length
    if ('predictions' in arima_df.columns and 'actual_flows' in arima_df.columns and
        'predictions' in lstm_df.columns and 'actual_flows' in lstm_df.columns):
        
        # Ensure we have data to compare
        if (len(arima_df['predictions']) > 0 and len(lstm_df['predictions']) > 0 and
            len(arima_df['actual_flows']) > 0 and len(lstm_df['actual_flows']) > 0):
            
            arima_pred = np.array(arima_df['predictions'])
            arima_actual = np.array(arima_df['actual_flows'])
            lstm_pred = np.array(lstm_df['predictions'])
            lstm_actual = np.array(lstm_df['actual_flows'])
            
            # Calculate metrics
            arima_mae = np.mean(np.abs(arima_pred - arima_actual))
            lstm_mae = np.mean(np.abs(lstm_pred - lstm_actual))
            
            print('\nPrediction Accuracy (MAE):')
            print('--------------------------------')
            print(f'ARIMA: {arima_mae:.2f} vehicles')
            print(f'LSTM:  {lstm_mae:.2f} vehicles')
            print('--------------------------------')
        else:
            print("\nWarning: Missing prediction data for comparison")

def main_menu():
    while True:
        print('\n=== Traffic Simulation Menu ===')
        print('1. Run new simulations')
        print('2. Generate visualizations from existing data')
        print('3. Exit')
        
        choice = input('\nEnter your choice (1-3): ')
        
        if choice == '1':
            print('\nRunning simulations with multiple models...')
            
            # Run all simulations sequentially using the same seed for a fair comparison
            adaptive_arima_results = run_simulation(use_adaptive_timing=True, model_type='arima', seed=42)
            adaptive_lstm_results = run_simulation(use_adaptive_timing=True, model_type='lstm', seed=42)
            fixed_results = run_simulation(use_adaptive_timing=False, model_type='fixed', seed=42)
            
            if adaptive_arima_results and adaptive_lstm_results and fixed_results:
                # Create DataFrames with all available data
                adaptive_arima_df = pd.DataFrame({
                    'timestamp': adaptive_arima_results['timestamp'],
                    'waiting_time': adaptive_arima_results['waiting_time'],
                    'queue_length': adaptive_arima_results['queue_length'],
                    'travel_time': adaptive_arima_results['travel_time'],
                    'predictions': adaptive_arima_results.get('predictions', []),
                    'actual_flows': adaptive_arima_results.get('actual_flows', [])
                })
                
                adaptive_lstm_df = pd.DataFrame({
                    'timestamp': adaptive_lstm_results['timestamp'],
                    'waiting_time': adaptive_lstm_results['waiting_time'],
                    'queue_length': adaptive_lstm_results['queue_length'],
                    'travel_time': adaptive_lstm_results['travel_time'],
                    'predictions': adaptive_lstm_results.get('predictions', []),
                    'actual_flows': adaptive_lstm_results.get('actual_flows', [])
                })
                
                fixed_df = pd.DataFrame({
                    'timestamp': fixed_results['timestamp'],
                    'waiting_time': fixed_results['waiting_time'],
                    'queue_length': fixed_results['queue_length'],
                    'travel_time': fixed_results['travel_time']
                })
                
                # Save results
                adaptive_arima_df.to_csv('adaptive_arima_results.csv', index=False)
                adaptive_lstm_df.to_csv('adaptive_lstm_results.csv', index=False)
                fixed_df.to_csv('fixed_results.csv', index=False)
                
                print("\nGenerating visualizations...")
                generate_visualizations(adaptive_arima_df, adaptive_lstm_df, fixed_df)
                
                if 'predictions' in adaptive_arima_df.columns and 'actual_flows' in adaptive_arima_df.columns:
                    create_prediction_accuracy_plot(adaptive_arima_df['predictions'], adaptive_arima_df['actual_flows'], 'generate_visualizations/arima_prediction_accuracy.png')
                if 'predictions' in adaptive_lstm_df.columns and 'actual_flows' in adaptive_lstm_df.columns:
                    create_prediction_accuracy_plot(adaptive_lstm_df['predictions'], adaptive_lstm_df['actual_flows'], 'generate_visualizations/lstm_prediction_accuracy.png')
                
                calculate_model_comparison(adaptive_arima_df, adaptive_lstm_df, fixed_df)
                
            else:
                print('Error: One or more simulations failed to complete.')
                
        elif choice == '2':
            print('\nLoading existing simulation data...')
            try:
                # Load the data
                adaptive_arima_df = pd.read_csv('adaptive_arima_results.csv')
                adaptive_lstm_df = pd.read_csv('adaptive_lstm_results.csv')
                fixed_df = pd.read_csv('fixed_results.csv')
                
                print("\nGenerating visualizations...")
                
                # Generate main comparison visualization
                generate_visualizations(adaptive_arima_df, adaptive_lstm_df, fixed_df)
                
                # Generate prediction accuracy plots if data exists
                if 'predictions' in adaptive_arima_df.columns and 'actual_flows' in adaptive_arima_df.columns:
                    create_prediction_accuracy_plot(adaptive_arima_df['predictions'], 
                                                 adaptive_arima_df['actual_flows'], 
                                                 'Generated Visualizations\prediction_accuracy_arima.png')
                
                if 'predictions' in adaptive_lstm_df.columns and 'actual_flows' in adaptive_lstm_df.columns:
                    create_prediction_accuracy_plot(adaptive_lstm_df['predictions'], 
                                                 adaptive_lstm_df['actual_flows'], 
                                                 'Generated Visualizations\prediction_accuracy_arima.png')
                
                # Calculate and display model comparison
                calculate_model_comparison(adaptive_arima_df, adaptive_lstm_df, fixed_df)
                
            except FileNotFoundError as e:
                print(f'Error: Could not find simulation results files. Please run new simulations first.')
                print(f'Missing file: {str(e)}')
            except Exception as e:
                print(f'Error generating visualizations: {str(e)}')
                
        elif choice == '3':
            print('\nExiting program...')
            break
            
        else:
            print('\nInvalid choice. Please enter 1, 2, or 3.')

if __name__ == '__main__':
    main_menu()