import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta, time
from pathlib import Path

# Translation dictionary
TRANSLATIONS = {
    'en': {
        'title': 'Production Scheduler',
        'optimization_weights': 'Optimization Weights',
        'batching_section': 'Batching Configuration',
        'batch_window': 'Time Window (Days)',
        'batch_window_help': 'Orders with same part number within this time window will be batched',
        'weights_info': 'Allocate weights to different optimization strategies. The sum must equal 100%',
        'makespan': 'Minimize Total Makespan (%)',
        'due_date': 'Prioritize Due Dates (%)',
        'utilization': 'Maximize Resource Utilization (%)',
        'setup_time': 'Minimize Setup Times (%)',
        'total': 'Total',
        'weights_error': 'Weights must sum to 100%',
        'upload_orders': 'Upload Production Orders (CSV)',
        'upload_resources': 'Upload Resources Data (CSV)',
        'generate_schedule': 'Generate Schedule',
        'schedule_generated': 'Production Schedule Generated!'
    },
    'zh': {
        'title': '生产排程系统',
        'optimization_weights': '优化权重',
        'batching_section': '批量计划设置',
        'batch_window': '时间范围 (天)',
        'batch_window_help': '在此时间范围内相同零件号的订单将被合并',
        'weights_info': '分配不同优化策略的权重。总和必须等于100%',
        'makespan': '最小化总生产时间 (%)',
        'due_date': '交期优先 (%)',
        'utilization': '最大化资源利用率 (%)',
        'setup_time': '最小化设置时间 (%)',
        'total': '总计',
        'weights_error': '权重总和必须等于100%',
        'upload_orders': '上传生产订单 (CSV)',
        'upload_resources': '上传资源数据 (CSV)',
        'generate_schedule': '生成排程',
        'schedule_generated': '排程已生成！'
    }
}

def init_session_state():
    if 'language' not in st.session_state:
        st.session_state.language = None

def get_text(key):
    return TRANSLATIONS[st.session_state.language][key]

def language_selector():
    if st.session_state.language is None:
        st.set_page_config(page_title="Production Scheduler", layout="wide")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.title("Language Selection / 语言选择")
            col_en, col_zh = st.columns(2)
            with col_en:
                if st.button("English", use_container_width=True):
                    st.session_state.language = 'en'
                    st.rerun()
            with col_zh:
                if st.button("中文", use_container_width=True):
                    st.session_state.language = 'zh'
                    st.rerun()
        return False
    return True

def calculate_batch_hours(batch):
    """Calculate total hours for a batch including all operations"""
    total_hours = 0
    for op_data in batch['operations'].values():
        total_hours += op_data['Run Time'] + op_data['Setup Time']
    return total_hours

def validate_weights(weights):
    """Validate that weights sum to 100%"""
    total = sum(weights.values())
    return abs(total - 100) < 0.01

def get_optimization_weights():
    """Get optimization weights and batching configuration"""
    st.sidebar.header(get_text('optimization_weights'))
    st.sidebar.info(get_text('weights_info'))
    
    weights = {
        'makespan': st.sidebar.number_input(get_text('makespan'),
                                        min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'due_date': st.sidebar.number_input(get_text('due_date'),
                                        min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'utilization': st.sidebar.number_input(get_text('utilization'),
                                           min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'setup_time': st.sidebar.number_input(get_text('setup_time'),
                                          min_value=0.0, max_value=100.0, value=25.0, step=5.0)
    }
    
    total = sum(weights.values())
    st.sidebar.write(f"{get_text('total')}: {total}%")
    
    if not validate_weights(weights):
        st.sidebar.error(get_text('weights_error'))
        return None
    
    # Add batching configuration
    st.sidebar.markdown("---")
    st.sidebar.header(get_text('batching_section'))
    
    # Add max batch hours configuration
    max_batch_hours = st.sidebar.number_input(
        "Maximum Batch Hours",
        min_value=1,
        max_value=500,
        value=150,
        help="Maximum allowed hours per batch to prevent excessive batch sizes."
    )
    
    batch_window = st.sidebar.number_input(
        get_text('batch_window'),
        min_value=0,
        max_value=30,
        value=5,
        help=get_text('batch_window_help')
    )
    
    # Return both weights and configurations
    result = {k: v/100.0 for k, v in weights.items()}
    result['batch_window'] = batch_window
    result['max_batch_hours'] = max_batch_hours
    return result

def schedule_operations(orders_df, resources_df):
    """Schedule operations considering finite capacity constraints and operation dependencies"""
    today = pd.Timestamp.now().normalize()
    
    # Initialize machine availability tracking
    machine_schedules = {}
    for _, resource in resources_df.iterrows():
        work_center = resource['WorkCenter']
        num_machines = int(resource['Available Quantity'])  # Ensure integer
        if num_machines > 0:
            machine_schedules[work_center] = [{
                'available_from': today,
                'machine_id': i + 1
            } for i in range(num_machines)]
    
    # Initialize job completion tracking
    job_completion = {}  # Track when each job's operations complete
    
    # Sort orders by priority and due date
    scheduled_orders = []
    orders_df = orders_df.sort_values(['JobPriority', 'Due Date'])
    
    # Group operations by job and sort by sequence
    for job_number, job_group in orders_df.groupby('Job Number'):
        job_operations = job_group.sort_values('operation sequence')
        prev_op_end = None
        
        # Process each operation in sequence
        for _, operation in job_operations.iterrows():
            work_center = operation['WorkCenter']
            
            # Skip if work center not found
            if work_center not in machine_schedules:
                print(f"Warning: Work center {work_center} not found in resources")
                continue
                
            total_hours = operation['Run Time'] + operation['Setup Time']
            
            # Find earliest available machine in the work center
            available_machines = machine_schedules[work_center]
            if not available_machines:
                print(f"Warning: No machines configured for work center {work_center}")
                continue
            
            # Sort machines by their available time
            available_machines.sort(key=lambda x: x['available_from'])
            
            # Select the first (earliest) available machine
            selected_machine = available_machines[0]
            
            # Determine earliest possible start time based on previous operation
            earliest_start = selected_machine['available_from']
            if prev_op_end:
                earliest_start = max(earliest_start, prev_op_end)
            
            # Schedule the operation
            start_time = earliest_start
            end_time = start_time + pd.Timedelta(hours=total_hours)
            due_date = pd.to_datetime(operation['Due Date'])
            
            # Check if we'll miss the due date
            if end_time > due_date:
                print(f"Warning: Operation for Job {operation['Job Number']} on {work_center} "
                      f"will finish at {end_time.strftime('%Y-%m-%d %H:%M')} which is after "
                      f"due date {due_date.strftime('%Y-%m-%d %H:%M')}")
            
            # Update machine availability
            selected_machine['available_from'] = end_time
            
            # Update previous operation end time for next operation in sequence
            prev_op_end = end_time
            
            # Create scheduled operation entry
            scheduled_op = operation.copy()
            scheduled_op['Start Time'] = start_time.strftime('%Y-%m-%d %H:%M')
            scheduled_op['Finish Time'] = end_time.strftime('%Y-%m-%d %H:%M')
            scheduled_op['Machine'] = f"{work_center}_M{selected_machine['machine_id']}"
            scheduled_op['Due Date Delay'] = max(0, (end_time - due_date).total_seconds() / 3600)  # delay in hours
            scheduled_orders.append(scheduled_op)
    
    result_df = pd.DataFrame(scheduled_orders)
    
    # Calculate schedule metrics
    if len(result_df) > 0:
        latest_end = pd.to_datetime(result_df['Finish Time']).max()
        earliest_start = pd.to_datetime(result_df['Start Time']).min()
        makespan = (latest_end - earliest_start).total_seconds() / 3600  # hours
        total_delay = result_df['Due Date Delay'].sum()
        print(f"\nSchedule Metrics:")
        print(f"Makespan: {makespan:.1f} hours")
        print(f"Total due date delay: {total_delay:.1f} hours")
    
    return result_df

def batch_and_schedule_orders(orders_df, resources_df, batch_window, max_batch_hours=150):
    """Batch orders and then schedule them with finite capacity"""
    if batch_window <= 0:
        return schedule_operations(orders_df, resources_df)
    
    # Sort orders by Part Number, Due Date, and operation sequence
    orders_df = orders_df.sort_values(['Part Number', 'Due Date', 'operation sequence'])
    processed_orders = []
    
    # Group by Part Number
    for part_number, group in orders_df.groupby('Part Number'):
        current_batch = None
        
        for _, order in group.iterrows():
            if current_batch is None:
                # Start new batch
                current_batch = {
                    'base': order.to_dict(),
                    'operations': {order['operation sequence']: order.to_dict()},
                    'Original Orders': [order['Job Number']]
                }
            elif (pd.to_datetime(order['Due Date']) - pd.to_datetime(current_batch['base']['Due Date'])).days <= batch_window:
                # Check if adding this order would exceed max_batch_hours
                new_batch = {
                    'operations': current_batch['operations'].copy(),
                    'Total Hours': 0
                }
                
                # Add the new operation or update existing one
                op_seq = order['operation sequence']
                if op_seq in new_batch['operations']:
                    new_batch['operations'][op_seq]['Run Time'] += order['Run Time']
                    new_batch['operations'][op_seq]['Setup Time'] = max(
                        new_batch['operations'][op_seq]['Setup Time'],
                        order['Setup Time']
                    )
                else:
                    new_batch['operations'][op_seq] = order.to_dict()
                
                # Calculate total hours for the potential new batch
                total_hours = calculate_batch_hours(new_batch)
                
                if total_hours <= max_batch_hours:
                    # Add to current batch since it's within size limit
                    current_batch['Original Orders'].append(order['Job Number'])
                    current_batch['base']['Quantity'] += order['Quantity']
                    current_batch['base']['JobPriority'] = min(current_batch['base']['JobPriority'], order['JobPriority'])
                    current_batch['base']['Due Date'] = min(current_batch['base']['Due Date'], order['Due Date'])
                    current_batch['operations'] = new_batch['operations']
                else:
                    # Current batch would exceed size limit, process it and start new batch
                    processed_orders.extend(process_batch(current_batch))
                    current_batch = {
                        'base': order.to_dict(),
                        'operations': {order['operation sequence']: order.to_dict()},
                        'Original Orders': [order['Job Number']]
                    }
            else:
                # Time window exceeded, process current batch and start new one
                processed_orders.extend(process_batch(current_batch))
                current_batch = {
                    'base': order.to_dict(),
                    'operations': {order['operation sequence']: order.to_dict()},
                    'Original Orders': [order['Job Number']]
                }
        
        # Process last batch for this part number
        if current_batch is not None:
            processed_orders.extend(process_batch(current_batch))
    
    # Convert batched orders to DataFrame and schedule them
    batched_df = pd.DataFrame(processed_orders)
    return schedule_operations(batched_df, resources_df)

def process_batch(batch):
    """Process a batch into individual entries"""
    processed_entries = []
    batch_job_number = f"BATCH_{batch['base']['Job Number']}"
    original_orders = ','.join(batch['Original Orders'])
    
    # Add each operation as a separate entry
    for op_seq, op_data in sorted(batch['operations'].items()):
        op_entry = op_data.copy()
        op_entry['Job Number'] = batch_job_number
        op_entry['Original Orders'] = original_orders
        
        # Calculate start and finish times
        finish_time = pd.to_datetime(op_entry['Due Date'])
        total_hours = op_entry['Run Time'] + op_entry['Setup Time']
        start_time = finish_time - pd.Timedelta(hours=total_hours)
        
        # Add times to the entry
        op_entry['Start Time'] = start_time.strftime('%Y-%m-%d %H:%M')
        op_entry['Finish Time'] = finish_time.strftime('%Y-%m-%d %H:%M')
        
        processed_entries.append(op_entry)
    
    return processed_entries

def load_production_orders(file):
    """Load and validate production orders CSV file"""
    try:
        df = None
        error_messages = []
        
        # Try different encodings
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
                # Reset file pointer to start
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding, on_bad_lines='skip')
                if not df.empty:
                    break
            except UnicodeDecodeError:
                error_messages.append(f"Failed to decode with {encoding} encoding")
                continue
            except pd.errors.EmptyDataError:
                error_messages.append(f"Empty file or no data found with {encoding} encoding")
                continue
            except Exception as e:
                error_messages.append(f"Error with {encoding} encoding: {str(e)}")
                continue
        
        if df is None or df.empty:
            raise ValueError(f"Could not load file with any encoding. Errors: {'; '.join(error_messages)}")
        
        # Print column names for debugging
        print("Loaded columns:", df.columns.tolist())
        print("Preview of first row:", df.iloc[0].to_dict() if not df.empty else "No data")
        
        # Check required columns
        required_columns = ['Job Number', 'Part Number', 'Due Date', 'operation sequence',
                         'WorkCenter', 'Place', 'Run Time', 'Setup Time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}\n"
                          f"Available columns are: {', '.join(df.columns)}")
        
        # Convert Due Date to datetime
        df['Due Date'] = pd.to_datetime(df['Due Date'].replace('#VALUE!', pd.NaT), format='%d/%m/%Y', errors='coerce')
        
        # Drop rows with invalid dates
        invalid_dates = df[df['Due Date'].isna()]
        if not invalid_dates.empty:
            st.warning(f"Found {len(invalid_dates)} rows with invalid dates. These orders will be skipped.")
            df = df.dropna(subset=['Due Date'])
        
        # Convert numeric columns
        df['Run Time'] = pd.to_numeric(df['Run Time'], errors='coerce')
        df['Setup Time'] = pd.to_numeric(df['Setup Time'], errors='coerce')
        
        # Drop any rows with invalid numeric values
        df = df.dropna(subset=['Run Time', 'Setup Time'])
        
        if df.empty:
            raise ValueError("No valid data remains after processing")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_resources(file):
    """Load and validate resources CSV file"""
    try:
        df = None
        error_messages = []
        
        # Try different encodings
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
                # Reset file pointer to start
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding, on_bad_lines='skip')
                if not df.empty:
                    break
            except UnicodeDecodeError:
                error_messages.append(f"Failed to decode with {encoding} encoding")
                continue
            except pd.errors.EmptyDataError:
                error_messages.append(f"Empty file or no data found with {encoding} encoding")
                continue
            except Exception as e:
                error_messages.append(f"Error with {encoding} encoding: {str(e)}")
                continue
        
        if df is None or df.empty:
            raise ValueError(f"Could not load resources file with any encoding. Errors: {'; '.join(error_messages)}")
            
        # Print column names for debugging
        print("Loaded resource columns:", df.columns.tolist())
        print("Preview of first resource row:", df.iloc[0].to_dict() if not df.empty else "No data")
        
        # Check required columns
        required_columns = ['WorkCenter', 'Place', 'Available Quantity', 'Shift hours']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}\n"
                          f"Available columns are: {', '.join(df.columns)}")
        
        # Convert numeric columns
        df['Available Quantity'] = pd.to_numeric(df['Available Quantity'], errors='coerce')
        df['Shift hours'] = pd.to_numeric(df['Shift hours'], errors='coerce')
        
        # Drop any rows with invalid numeric values
        df = df.dropna(subset=['Available Quantity', 'Shift hours'])
        
        if df.empty:
            raise ValueError("No valid resource data remains after processing")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading resources file: {str(e)}")
        return None

def create_gantt_chart(df, selected_job=None):
    """Create a Gantt chart for the production schedule with optional job highlighting"""
    # Convert datetime strings to datetime objects
    df = df.copy()
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['Finish Time'] = pd.to_datetime(df['Finish Time'])
    
    # Sort by WorkCenter and Start Time
    df_sorted = df.sort_values(['WorkCenter', 'Start Time'])
    
    # Create figure
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    color_map = {}
    color_idx = 0
    
    # Add bars for each operation
    for idx, row in df_sorted.iterrows():
        if row['WorkCenter'] not in color_map:
            color_map[row['WorkCenter']] = colors[color_idx % len(colors)]
            color_idx += 1
        
        # Determine bar color and opacity based on selection
        if selected_job:
            is_selected = row['Job Number'] == selected_job
            color = color_map[row['WorkCenter']]
            opacity = 1.0 if is_selected else 0.3
            line_width = 2 if is_selected else 0
            line_color = 'black' if is_selected else None
        else:
            color = color_map[row['WorkCenter']]
            opacity = 1.0
            line_width = 0
            line_color = None
        
        # Add task bar
        fig.add_trace(go.Bar(
            x=[row['Start Time'], row['Finish Time']],
            y=[row['WorkCenter']],
            orientation='h',
            marker=dict(
                color=color,
                opacity=opacity,
                line=dict(
                    width=line_width,
                    color=line_color
                )
            ),
            hovertext=(f"Job: {row['Job Number']}<br>"
                      f"Part: {row['Part Number']}<br>"
                      f"Operation: {row['operation sequence']}<br>"
                      f"Setup: {row['Setup Time']}h<br>"
                      f"Run: {row['Run Time']}h<br>"
                      f"Start: {row['Start Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                      f"End: {row['Finish Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                      f"Machine: {row['Machine']}"),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Production Schedule by Work Center",
        xaxis_title="Time",
        yaxis_title="Work Center",
        height=400 + len(df['WorkCenter'].unique()) * 40,  # Adjusted for work center level
        barmode='overlay',
        bargap=0.2,  # Increased gap for better readability
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(
            tickmode='linear',
            type='category'
        )
    )
    
    return fig

def create_workload_heatmap(orders_df, resources_df):
    """Create a heatmap showing work center utilization across dates"""
    # Convert datetime strings to datetime objects
    orders_df = orders_df.copy()
    orders_df['Start Time'] = pd.to_datetime(orders_df['Start Time'])
    orders_df['Finish Time'] = pd.to_datetime(orders_df['Finish Time'])
    
    # Create a date range covering the entire schedule
    date_range = pd.date_range(
        start=orders_df['Start Time'].min().normalize(),
        end=orders_df['Finish Time'].max().normalize(),
        freq='D'
    )
    
    # Initialize the workload calculation
    workload_data = []
    
    # Process each operation
    for _, operation in orders_df.iterrows():
        # Get the operation dates
        op_dates = pd.date_range(
            start=operation['Start Time'].normalize(),
            end=operation['Finish Time'].normalize(),
            freq='D'
        )
        
        # Calculate hours per day for this operation
        for op_date in op_dates:
            start = max(operation['Start Time'], pd.Timestamp(op_date))
            end = min(operation['Finish Time'], pd.Timestamp(op_date) + pd.Timedelta(days=1))
            hours = (end - start).total_seconds() / 3600
            
            workload_data.append({
                'WorkCenter': operation['WorkCenter'],
                'Date': op_date,
                'Hours': hours
            })
    
    # Convert to DataFrame
    workload = pd.DataFrame(workload_data)
    
    # Sum hours by work center and date
    workload = workload.groupby(['WorkCenter', 'Date'])['Hours'].sum().reset_index()
    
    # Create full date range for all work centers
    all_workcenters = orders_df['WorkCenter'].unique()
    full_index = pd.MultiIndex.from_product(
        [all_workcenters, date_range],
        names=['WorkCenter', 'Date']
    )
    
    # Reindex to include all dates, fill missing values with 0
    workload = workload.set_index(['WorkCenter', 'Date']).reindex(full_index, fill_value=0).reset_index()
    
    # Merge with resources to get shift hours and number of machines
    workload = pd.merge(
        workload,
        resources_df[['WorkCenter', 'Shift hours', 'Available Quantity']],
        on='WorkCenter',
        how='left'
    )
    
    # Calculate utilization percentage (hours used / total available hours per day)
    workload['Total Available Hours'] = workload['Shift hours'] * workload['Available Quantity']
    workload['Utilization'] = (workload['Hours'] / workload['Total Available Hours'] * 100).clip(0, 100)
    
    # Pivot data for heatmap
    workload_pivot = workload.pivot_table(
        index='WorkCenter',
        columns='Date',
        values='Utilization',
        fill_value=0
    )
    
    # Sort work centers
    workload_pivot = workload_pivot.reindex(sorted(workload_pivot.index))
    
    # Create heatmap with utilization percentages
    fig = go.Figure(data=go.Heatmap(
        z=workload_pivot.values.tolist(),
        x=workload_pivot.columns.strftime('%Y-%m-%d'),
        y=workload_pivot.index,
        colorscale=[
            [0, 'lightblue'],     # 0% utilization
            [0.4, 'lightgreen'],  # 40% utilization
            [0.7, 'yellow'],      # 70% utilization
            [0.9, 'orange'],      # 90% utilization
            [1.0, 'red']          # 100% utilization
        ],
        text=[[f"{val:.1f}%" for val in row] for row in workload_pivot.values],
        texttemplate="%{text}",
        hovertemplate='Work Center: %{y}<br>Date: %{x}<br>Utilization: %{z:.1f}%<br>',
        colorbar=dict(
            title='Utilization %',
            ticksuffix='%'
        ),
        zmin=0,
        zmax=100
    ))
    
    # Update layout
    fig.update_layout(
        title="Work Center Utilization Heatmap",
        xaxis_title="Date",
        yaxis_title="Work Center",
        height=400 + len(workload_pivot.index) * 40,  # Increased height per row
        yaxis=dict(
            tickmode='linear',  # Show all work center names
            type='category'
        ),
        margin=dict(t=50, l=200)  # Increased left margin for work center names
    )
    
    return fig

def main():
    init_session_state()
    if not language_selector():
        return

    st.title(get_text('title'))
    
    # Get optimization weights and batch settings
    settings = get_optimization_weights()
    if not settings:
        return

    col1, col2 = st.columns(2)
    with col1:
        orders_file = st.file_uploader(get_text('upload_orders'), type=['csv'])
        if orders_file:
            orders_df = load_production_orders(orders_file)
            if orders_df is not None:
                st.write("Production Orders Preview:")
                st.dataframe(orders_df.head())
                st.info(f"Total orders: {len(orders_df)}")
                
    with col2:
        resources_file = st.file_uploader(get_text('upload_resources'), type=['csv'])
        if resources_file:
            resources_df = load_resources(resources_file)
            if resources_df is not None:
                st.write("Resources Preview:")
                st.dataframe(resources_df)
                st.info(f"Total resources: {len(resources_df)}")

    if orders_file and resources_file:
        orders_df = load_production_orders(orders_file)
        resources_df = load_resources(resources_file)
        
        if orders_df is not None and resources_df is not None and st.button(get_text('generate_schedule')):
            with st.spinner("Generating schedule..."):
                try:
                    # Capture warnings in a StringIO buffer
                    import io
                    import sys
                    warning_output = io.StringIO()
                    sys.stdout = warning_output
                    
                    # Apply batching and scheduling with finite capacity
                    scheduled_orders = batch_and_schedule_orders(
                        orders_df,
                        resources_df,
                        settings['batch_window'],
                        settings['max_batch_hours']
                    )
                    
                    # Restore stdout and get warnings
                    sys.stdout = sys.__stdout__
                    warnings = warning_output.getvalue()
                    
                    if scheduled_orders is None or len(scheduled_orders) == 0:
                        st.error("No operations could be scheduled. Please check resource availability.")
                        return
                        
                    # Display any warnings that occurred during scheduling
                    if warnings.strip():
                        with st.expander("Show Scheduling Warnings", expanded=False):
                            st.warning(warnings)
                except Exception as e:
                    st.error(f"Error during scheduling: {str(e)}")
                    return
                
                # Calculate and display metrics
                original_orders = len(orders_df['Job Number'].unique())
                scheduled_orders_count = len(scheduled_orders['Job Number'].unique())
                avg_batch_size = scheduled_orders.groupby('Job Number')['Run Time'].sum().mean()
                
                st.success(get_text('schedule_generated'))
                
                # Display schedule metrics in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Batch Statistics:**")
                    st.write(f"""
                    - Original orders: {original_orders}
                    - Scheduled orders: {scheduled_orders_count}
                    - Average batch size: {avg_batch_size:.1f} hours
                    """)
                
                with col2:
                    # Calculate schedule performance metrics
                    makespan = (pd.to_datetime(scheduled_orders['Finish Time']).max() -
                              pd.to_datetime(scheduled_orders['Start Time']).min()).total_seconds() / 3600
                    delayed_jobs = scheduled_orders[pd.to_datetime(scheduled_orders['Finish Time']) >
                                                 pd.to_datetime(scheduled_orders['Due Date'])]
                    
                    st.write("**Schedule Performance:**")
                    st.write(f"""
                    - Total makespan: {makespan:.1f} hours
                    - Delayed jobs: {len(delayed_jobs)}
                    - Average delay: {delayed_jobs['Due Date Delay'].mean():.1f} hours
                    """)
                
                st.subheader(get_text('title'))
                
                # Reorder columns to show timing information and machine assignment first
                cols = ['Job Number', 'Part Number', 'Start Time', 'Finish Time', 'WorkCenter', 'Machine',
                       'Run Time', 'Setup Time', 'operation sequence', 'Place', 'Original Orders']
                display_df = scheduled_orders[cols]
                
                # Display schedule with reordered columns
                st.dataframe(display_df)
                
                # Create visualizations
                st.subheader("Schedule Visualizations")
                
                # Add job selection dropdown
                unique_jobs = sorted(scheduled_orders['Job Number'].unique())
                job_col1, job_col2 = st.columns([2, 1])
                
                with job_col1:
                    selected_job = st.selectbox(
                        "Select a job to highlight its operations",
                        ["All Jobs"] + list(unique_jobs),
                        format_func=lambda x: f"Job {x}" if x != "All Jobs" else x
                    )
                
                # Convert selection to actual job number
                job_to_highlight = None if selected_job == "All Jobs" else selected_job
                
                # Show job details if a specific job is selected
                if job_to_highlight:
                    job_ops = scheduled_orders[scheduled_orders['Job Number'] == job_to_highlight].copy()
                    job_ops['Start Time'] = pd.to_datetime(job_ops['Start Time'])
                    job_ops['Finish Time'] = pd.to_datetime(job_ops['Finish Time'])
                    total_duration = (job_ops['Finish Time'].max() - job_ops['Start Time'].min()).total_seconds() / 3600
                    
                    with job_col2:
                        st.write("**Job Details:**")
                        st.write(f"Part Number: {job_ops.iloc[0]['Part Number']}")
                        st.write(f"Number of Operations: {len(job_ops)}")
                        st.write(f"Total Processing Time: {job_ops['Run Time'].sum():.1f}h")
                        st.write(f"Total Setup Time: {job_ops['Setup Time'].sum():.1f}h")
                        st.write(f"Total Duration: {total_duration:.1f}h")
                
                # Create Gantt chart with job highlighting
                fig_gantt = create_gantt_chart(scheduled_orders, job_to_highlight)
                st.plotly_chart(fig_gantt, use_container_width=True)
                
                # Create heatmap
                fig_heatmap = create_workload_heatmap(scheduled_orders, resources_df)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Download option
                csv = scheduled_orders.to_csv(index=False)
                st.download_button(
                    "Download Schedule",
                    csv,
                    "production_schedule.csv",
                    "text/csv",
                    key='download-csv'
                )

if __name__ == "__main__":
    main()