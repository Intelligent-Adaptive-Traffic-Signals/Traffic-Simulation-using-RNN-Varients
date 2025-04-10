import traci
import numpy as np
import pandas as pd
import pytz
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sumolib.net import Phase

def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
    return currentDT.strftime('%Y-%m-%d %H:%M:%S')

def run_simulation(use_adaptive_timing=True, seed=42):
    sumoCmd = ['sumo-gui', '-c', 'osm.sumocfg', '--seed', str(seed)]
    if not use_adaptive_timing:
        sumoCmd = ['sumo-gui', '-c', 'osm.sumocfg', '--seed', str(seed+1)]
    
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
    
    time_series_data = []
    time_stamps = []
    waiting_times = []
    queue_lengths = []
    travel_times = []
    throughput = []
    predictions = []
    actual_flows = []
    
    completed_vehicles = set()
    vehicle_start_times = {}
    lane_densities = {}
    incoming_lanes = []
    completed_travel_times = []
    
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
        
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)
        time_series_data.append(vehicle_count)
        time_stamps.append(current_time)
        
        for veh_id in vehicle_ids:
            if veh_id not in vehicle_start_times:
                vehicle_start_times[veh_id] = step
        
        total_waiting_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids) if vehicle_ids else 0
        avg_waiting_time = total_waiting_time / vehicle_count if vehicle_count > 0 else 0
        waiting_times.append(avg_waiting_time)
        
        queue = sum(1 for veh_id in vehicle_ids if traci.vehicle.getSpeed(veh_id) < 0.1)
        queue_lengths.append(queue)
        
        arrived = traci.simulation.getArrivedNumber()
        throughput.append(arrived)
        
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
        
        if use_adaptive_timing and len(time_series_data) > 30 and step % 5 == 0:
            try:
                actual_flow = vehicle_count
                transformed_data = np.log1p(time_series_data[-30:])
                model = ARIMA(transformed_data, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                prediction = np.expm1(model_fit.forecast(steps=5))
                predicted_traffic = max(prediction[-1], 0)
                predictions.append(predicted_traffic)
                actual_flows.append(actual_flow)
                
                print(f'Predicted Traffic Flow: {predicted_traffic:.2f}, Actual: {actual_flow}')
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
                print(f'Updated Traffic Light {target_tl} timings based on lane congestion.')
                
            except Exception as e:
                print(f'ARIMA Training Error: {e}')
    
    traci.close()
    
    results = {
        'timestamp': time_stamps,
        'vehicle_count': time_series_data,
        'waiting_time': waiting_times,
        'queue_length': queue_lengths,
        'travel_time': travel_times,
        'predictions': predictions,
        'actual_flows': actual_flows,
        'lane_densities': lane_densities
    }
    
    return results

def generate_visualizations(adaptive_df, fixed_df):
    plt.style.use('bmh')
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Traffic Control System Comparison', fontsize=16, y=0.95)
    
    colors = {
        'adaptive': '#2ecc71',
        'fixed': '#e74c3c'
    }
    line_styles = {
        'adaptive': '-',
        'fixed': '--'
    }
    
    def style_axis(ax, title, xlabel, ylabel):
        ax.set_title(title, pad=20, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, linestyle=':', alpha=0.6)
    
    window = 30
    adaptive_wait_smooth = adaptive_df['waiting_time'].rolling(window=window).mean()
    fixed_wait_smooth = fixed_df['waiting_time'].rolling(window=window).mean()
    
    axes[0, 0].plot(adaptive_wait_smooth, color=colors['adaptive'],
                    linestyle=line_styles['adaptive'], linewidth=2, label='Adaptive Timing')
    axes[0, 0].plot(fixed_wait_smooth, color=colors['fixed'],
                    linestyle=line_styles['fixed'], linewidth=2, label='Fixed Timing')
    style_axis(axes[0, 0], 'Average Waiting Time\n(30-step rolling average)',
              'Simulation Step', 'Waiting Time (seconds)')
    
    adaptive_queue_smooth = adaptive_df['queue_length'].rolling(window=window).mean()
    fixed_queue_smooth = fixed_df['queue_length'].rolling(window=window).mean()
    
    axes[0, 1].plot(adaptive_queue_smooth, color=colors['adaptive'],
                    linestyle=line_styles['adaptive'], linewidth=2, label='Adaptive Timing')
    axes[0, 1].plot(fixed_queue_smooth, color=colors['fixed'],
                    linestyle=line_styles['fixed'], linewidth=2, label='Fixed Timing')
    style_axis(axes[0, 1], 'Queue Length\n(30-step rolling average)',
              'Simulation Step', 'Number of Vehicles')
    
    adaptive_travel_smooth = adaptive_df['travel_time'].rolling(window=window).mean()
    fixed_travel_smooth = fixed_df['travel_time'].rolling(window=window).mean()
    
    axes[1, 0].plot(adaptive_travel_smooth, color=colors['adaptive'],
                    linestyle=line_styles['adaptive'], linewidth=2, label='Adaptive Timing')
    axes[1, 0].plot(fixed_travel_smooth, color=colors['fixed'],
                    linestyle=line_styles['fixed'], linewidth=2, label='Fixed Timing')
    style_axis(axes[1, 0], 'Travel Time\n(30-step rolling average)',
              'Simulation Step', 'Travel Time (seconds)')
    
    axes[1, 1].remove()
    axes[1, 1] = fig.add_subplot(224)
    
    
    adaptive_wait = np.mean(adaptive_df['waiting_time'])
    fixed_wait = np.mean(fixed_df['waiting_time'])
    wait_improvement = ((fixed_wait - adaptive_wait) / fixed_wait) * 100 if fixed_wait > 0 else 0
    
    adaptive_queue = np.mean(adaptive_df['queue_length'])
    fixed_queue = np.mean(fixed_df['queue_length'])
    queue_improvement = ((fixed_queue - adaptive_queue) / fixed_queue) * 100 if fixed_queue > 0 else 0
    
    adaptive_travel = np.mean(adaptive_df['travel_time'])
    fixed_travel = np.mean(fixed_df['travel_time'])
    travel_improvement = ((fixed_travel - adaptive_travel) / fixed_travel) * 100 if fixed_travel > 0 else 0
    
 
    wait_pct = ((fixed_wait_smooth - adaptive_wait_smooth) / fixed_wait_smooth * 100).fillna(0)
    queue_pct = ((fixed_queue_smooth - adaptive_queue_smooth) / fixed_queue_smooth * 100).fillna(0)
    travel_pct = ((fixed_travel_smooth - adaptive_travel_smooth) / fixed_travel_smooth * 100).fillna(0)
    
    axes[1, 1].plot(wait_pct, color='#3498db', linewidth=2, label='Waiting Time')
    axes[1, 1].plot(queue_pct, color='#e67e22', linewidth=2, label='Queue Length')
    axes[1, 1].plot(travel_pct, color='#9b59b6', linewidth=2, label='Travel Time')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    style_axis(axes[1, 1], 'Percentage Improvement\n(Adaptive vs Fixed)',
              'Simulation Step', 'Improvement (%)')
    
    
    text = f'Average Improvements:\nWaiting Time: {wait_improvement:.1f}%\nQueue Length: {queue_improvement:.1f}%\nTravel Time: {travel_improvement:.1f}%'
    axes[1, 1].text(0.02, 0.98, text, transform=axes[1, 1].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ARIMA_Traffic_Control.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('\nVisualizations have been generated and saved.')

def calculate_and_print_metrics(adaptive_df, fixed_df):
   
    adaptive_wait = np.mean(adaptive_df['waiting_time'])
    fixed_wait = np.mean(fixed_df['waiting_time'])
    wait_improvement = ((fixed_wait - adaptive_wait) / fixed_wait) * 100 if fixed_wait > 0 else 0
    print(f'\nAverage Waiting Time:')
    print(f'Adaptive = {adaptive_wait:.2f}s')
    print(f'Fixed = {fixed_wait:.2f}s')
    print(f'Improvement = {wait_improvement:.2f}%')
    
   
    adaptive_queue = np.mean(adaptive_df['queue_length'])
    fixed_queue = np.mean(fixed_df['queue_length'])
    queue_improvement = ((fixed_queue - adaptive_queue) / fixed_queue) * 100 if fixed_queue > 0 else 0
    print(f'\nAverage Queue Length:')
    print(f'Adaptive = {adaptive_queue:.2f} vehicles')
    print(f'Fixed = {fixed_queue:.2f} vehicles')
    print(f'Improvement = {queue_improvement:.2f}%')
    
    
    if 'travel_time' in adaptive_df.columns and 'travel_time' in fixed_df.columns:
        adaptive_travel = np.mean(adaptive_df['travel_time'])
        fixed_travel = np.mean(fixed_df['travel_time'])
        travel_improvement = ((fixed_travel - adaptive_travel) / fixed_travel) * 100 if fixed_travel > 0 else 0
        print(f'\nAverage Travel Time:')
        print(f'Adaptive = {adaptive_travel:.2f}s')
        print(f'Fixed = {fixed_travel:.2f}s')
        print(f'Improvement = {travel_improvement:.2f}%')

def main_menu():
    while True:
        print('\n=== Traffic Simulation Menu ===')
        print('1. Run new simulations')
        print('2. Generate visualizations from existing data')
        print('3. Exit')
        
        choice = input('\nEnter your choice (1-3): ')
        
        if choice == '1':
            print('\nRunning new simulations...')
            adaptive_results = run_simulation(use_adaptive_timing=True, seed=42)
            fixed_results = run_simulation(use_adaptive_timing=False, seed=42)
            
            if adaptive_results and fixed_results:
                adaptive_df = pd.DataFrame({
                    'timestamp': adaptive_results['timestamp'],
                    'waiting_time': adaptive_results['waiting_time'],
                    'queue_length': adaptive_results['queue_length'],
                    'travel_time': adaptive_results['travel_time']
                })
                
                fixed_df = pd.DataFrame({
                    'timestamp': fixed_results['timestamp'],
                    'waiting_time': fixed_results['waiting_time'],
                    'queue_length': fixed_results['queue_length'],
                    'travel_time': fixed_results['travel_time']
                })
                
              
                calculate_and_print_metrics(adaptive_df, fixed_df)
                
             
                
                
                
                adaptive_df.to_csv('adaptive_traffic_data.csv', index=False)
                fixed_df.to_csv('fixed_traffic_data.csv', index=False)
                
                generate_visualizations(adaptive_df, fixed_df)
                print('\nSimulations completed and results saved.')
            else:
                print('Error: One or both simulations failed to run properly.')
        
        elif choice == '2':
            try:
                print('\nLoading existing data...')
                adaptive_df = pd.read_csv('adaptive_traffic_data.csv')
                fixed_df = pd.read_csv('fixed_traffic_data.csv')
                
                print('\nTraffic Efficiency Metrics:')
                calculate_and_print_metrics(adaptive_df, fixed_df)
                
                generate_visualizations(adaptive_df, fixed_df)
            except FileNotFoundError:
                print('\nError: Could not find the required CSV files.')
                print('Please run the simulations first to generate the data files.')
            except KeyError as e:
                print(f'\nError: Missing data column {e}.')
                print('Please run new simulations to generate complete data.')
        
        elif choice == '3':
            print('\nExiting program...')
            break
        
        else:
            print('\nInvalid choice. Please enter 1, 2, or 3.')

if __name__ == '__main__':
    main_menu()