import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta, time
from pathlib import Path

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
    st.sidebar.header('Optimization Weights')
    st.sidebar.info('Allocate weights to different optimization strategies. The sum must equal 100%')
    
    weights = {
        'makespan': st.sidebar.number_input('Minimize Total Makespan (%)',
                                        min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'due_date': st.sidebar.number_input('Prioritize Due Dates (%)',
                                        min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'utilization': st.sidebar.number_input('Maximize Resource Utilization (%)',
                                           min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'setup_time': st.sidebar.number_input('Minimize Setup Times (%)',
                                          min_value=0.0, max_value=100.0, value=25.0, step=5.0)
    }
    
    total = sum(weights.values())
    st.sidebar.write(f"Total: {total}%")
    
    if abs(total - 100) > 0.01:
        st.sidebar.error('Weights must sum to 100%')
        return None
    
    st.sidebar.markdown("---")
    st.sidebar.header('Batching Configuration')
    
    max_batch_hours = st.sidebar.number_input(
        "Maximum Batch Hours",
        min_value=1,
        max_value=500,
        value=150,
        help="Maximum allowed hours per batch to prevent excessive batch sizes."
    )
    
    batch_window = st.sidebar.number_input(
        'Time Window (Days)',
        min_value=0,
        max_value=30,
        value=5,
        help='Orders with same part number within this time window will be batched'
    )
    
    result = {k: v/100.0 for k, v in weights.items()}
    result['batch_window'] = batch_window
    result['max_batch_hours'] = max_batch_hours
    return result

def load_production_orders(file):
    """Load and validate production orders CSV file"""
    try:
        df = None
        error_messages = []
        
        # Try different encodings
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
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
                          'WorkCenter', 'Place', 'Run Time', 'Setup Time', 'JobPriority']
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
        df['JobPriority'] = pd.to_numeric(df['JobPriority'], errors='coerce')
        
        # Drop any rows with invalid numeric values
        df = df.dropna(subset=['Run Time', 'Setup Time', 'JobPriority'])
        
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

def create_batches(orders_df, max_batch_hours):
    """Create batches of operations optimizing for setup time"""
    batches = []
    
    # Group operations by Part Number and WorkCenter
    grouped_ops = orders_df.groupby(['Part Number', 'WorkCenter'])
    
    for (part_number, work_center), group in grouped_ops:
        # Sort operations by priority and due date
        group = group.sort_values(['JobPriority', 'Due Date'])
        
        current_batch = []
        batch_run_time = 0
        batch_setup_time = 0
        
        for _, operation in group.iterrows():
            # For new batch, consider full setup time
            if not current_batch:
                batch_setup_time = operation['Setup Time']
                batch_run_time = operation['Run Time']
                current_batch.append(operation)
            else:
                # For existing batch, only add run time (setup already counted)
                if batch_run_time + operation['Run Time'] <= max_batch_hours:
                    batch_run_time += operation['Run Time']
                    batch_setup_time = max(batch_setup_time, operation['Setup Time'])
                    current_batch.append(operation)
                else:
                    # Finalize current batch
                    batches.append({
                        'operations': current_batch,
                        'work_center': work_center,
                        'part_number': part_number,
                        'total_hours': batch_run_time + batch_setup_time,
                        'setup_time': batch_setup_time,
                        'priority': min(op['JobPriority'] for op in current_batch)
                    })
                    # Start new batch with current operation
                    current_batch = [operation]
                    batch_run_time = operation['Run Time']
                    batch_setup_time = operation['Setup Time']
        
        # Add final batch
        if current_batch:
            batches.append({
                'operations': current_batch,
                'work_center': work_center,
                'part_number': part_number,
                'total_hours': batch_run_time + batch_setup_time,
                'setup_time': batch_setup_time,
                'priority': min(op['JobPriority'] for op in current_batch)
            })
    
    # Sort batches by priority and due date
    batches.sort(key=lambda x: (x['priority'], min(op['Due Date'] for op in x['operations'])))
    return batches

def schedule_operations(orders_df, resources_df, settings):
    """Schedule operations with optimized batching and priority-based forwarding"""
    today = pd.Timestamp.now().normalize()
    
    # Initialize machine availability tracking
    machine_schedules = {}
    for _, resource in resources_df.iterrows():
        work_center = resource['WorkCenter']
        num_machines = int(resource['Available Quantity'])
        if num_machines > 0:
            machine_schedules[work_center] = [{
                'available_from': today,
                'machine_id': i + 1,
                'total_hours': 0
            } for i in range(num_machines)]

    # Create optimized batches
    batches = create_batches(orders_df, settings['max_batch_hours'])
    
    # Track completion times for each job's operations
    job_completion_times = {}
    scheduled_orders = []

    # Process each batch while respecting job dependencies
    for batch in batches:
        work_center = batch['work_center']
        machines = machine_schedules[work_center]
        
        # Find earliest available machine considering job dependencies
        earliest_start = None
        selected_machine = None
        
        for machine in machines:
            possible_start = machine['available_from']
            
            # Check dependencies for all operations in batch
            for operation in batch['operations']:
                job_number = operation['Job Number']
                op_seq = operation['operation sequence']
                
                # If job has previous operations, consider their completion times
                if job_number in job_completion_times:
                    prev_op_time = job_completion_times[job_number].get(op_seq - 1)
                    if prev_op_time:
                        possible_start = max(possible_start, prev_op_time)
            
            if earliest_start is None or possible_start < earliest_start:
                earliest_start = possible_start
                selected_machine = machine
        
        if selected_machine is None:
            print(f"Warning: No machine available for {work_center}")
            continue
        
        # Schedule all operations in the batch
        current_time = earliest_start
        
        # Apply setup time once for the batch
        setup_end = current_time + pd.Timedelta(hours=batch['setup_time'])
        current_time = setup_end
        
        # Schedule each operation in the batch
        for operation in batch['operations']:
            # Only use run time since setup is already accounted for
            run_time = operation['Run Time']
            end_time = current_time + pd.Timedelta(hours=run_time)
            
            # Record scheduled operation
            scheduled_op = operation.copy()
            scheduled_op['Start Time'] = current_time.strftime('%Y-%m-%d %H:%M')
            scheduled_op['Finish Time'] = end_time.strftime('%Y-%m-%d %H:%M')
            scheduled_op['Machine'] = f"{work_center}_M{selected_machine['machine_id']}"
            scheduled_orders.append(scheduled_op)
            
            # Update job completion tracking
            job_number = operation['Job Number']
            op_seq = operation['operation sequence']
            if job_number not in job_completion_times:
                job_completion_times[job_number] = {}
            job_completion_times[job_number][op_seq] = end_time
            
            # Move to next operation time
            current_time = end_time
        
        # Update machine availability
        selected_machine['available_from'] = current_time
        selected_machine['total_hours'] += batch['total_hours']
    
    result_df = pd.DataFrame(scheduled_orders)
    
    if len(result_df) > 0:
        # Convert datetime strings to datetime objects for calculations
        result_df['Start Time'] = pd.to_datetime(result_df['Start Time'])
        result_df['Finish Time'] = pd.to_datetime(result_df['Finish Time'])
        
        # Calculate schedule metrics
        latest_end = result_df['Finish Time'].max()
        earliest_start = result_df['Start Time'].min()
        makespan = (latest_end - earliest_start).total_seconds() / 3600
        
        print(f"\nSchedule Metrics:")
        print(f"Makespan: {makespan:.1f} hours")
        print(f"Schedule spans from {earliest_start.strftime('%Y-%m-%d')} to {latest_end.strftime('%Y-%m-%d')}")
        
        # Calculate and print batching metrics
        total_batches = len(batches)
        avg_batch_size = sum(len(batch['operations']) for batch in batches) / total_batches
        total_setup_time = sum(batch['setup_time'] for batch in batches)
        setup_time_saved = sum(
            sum(op['Setup Time'] for op in batch['operations']) - batch['setup_time']
            for batch in batches
        )
        
        print(f"\nBatching Metrics:")
        print(f"Total batches: {total_batches}")
        print(f"Average batch size: {avg_batch_size:.1f} operations")
        print(f"Total setup time: {total_setup_time:.1f} hours")
        print(f"Setup time saved: {setup_time_saved:.1f} hours")
        
        # Convert back to string format for display
        result_df['Start Time'] = result_df['Start Time'].dt.strftime('%Y-%m-%d %H:%M')
        result_df['Finish Time'] = result_df['Finish Time'].dt.strftime('%Y-%m-%d %H:%M')
    
    return result_df

def create_gantt_chart(df, selected_job=None):
    """Create a Gantt chart for the production schedule with optional job highlighting"""
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
        height=400 + len(df['WorkCenter'].unique()) * 40,
        barmode='overlay',
        bargap=0.2,
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
    orders_df = orders_df.copy()
    orders_df['Start Time'] = pd.to_datetime(orders_df['Start Time'])
    orders_df['Finish Time'] = pd.to_datetime(orders_df['Finish Time'])
    
    # Create a date range covering the entire schedule
    date_range = pd.date_range(
        start=orders_df['Start Time'].min().normalize(),
        end=orders_df['Finish Time'].max().normalize(),
        freq='D'
    )
    
    # Initialize workload calculation
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
    
    # Create heatmap
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
        height=400 + len(workload_pivot.index) * 40,
        yaxis=dict(
            tickmode='linear',
            type='category'
        ),
        margin=dict(t=50, l=200)
    )
    
    return fig

def main():
    st.set_page_config(page_title="Production Scheduler", layout="wide")
    st.title("Production Scheduler")
    
    # Get optimization weights and batch settings
    settings = get_optimization_weights()
    if not settings:
        return

    col1, col2 = st.columns(2)
    with col1:
        orders_file = st.file_uploader("Upload Production Orders (CSV)", type=['csv'])
        if orders_file:
            orders_df = load_production_orders(orders_file)
            if orders_df is not None:
                st.write("Production Orders Preview:")
                st.dataframe(orders_df.head())
                st.info(f"Total orders: {len(orders_df)}")
                
    with col2:
        resources_file = st.file_uploader("Upload Resources Data (CSV)", type=['csv'])
        if resources_file:
            resources_df = load_resources(resources_file)
            if resources_df is not None:
                st.write("Resources Preview:")
                st.dataframe(resources_df)
                st.info(f"Total resources: {len(resources_df)}")

    if orders_file and resources_file:
        orders_df = load_production_orders(orders_file)
        resources_df = load_resources(resources_file)
        
        if orders_df is not None and resources_df is not None and st.button("Generate Schedule"):
            with st.spinner("Generating schedule..."):
                try:
                    # Capture warnings in a StringIO buffer
                    import io
                    import sys
                    warning_output = io.StringIO()
                    sys.stdout = warning_output
                    
                    # Apply scheduling with finite capacity and batching configuration
                    scheduled_orders = schedule_operations(orders_df, resources_df, settings)
                    
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
                
                st.success("Schedule generated!")
                
                # Calculate and display metrics
                original_orders = len(orders_df['Job Number'].unique())
                scheduled_orders_count = len(scheduled_orders['Job Number'].unique())
                avg_run_time = scheduled_orders['Run Time'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Schedule Statistics:**")
                    st.write(f"""
                    - Original orders: {original_orders}
                    - Scheduled orders: {scheduled_orders_count}
                    - Average run time: {avg_run_time:.1f} hours
                    """)
                
                with col2:
                    # Calculate schedule performance metrics
                    scheduled_orders['Start Time'] = pd.to_datetime(scheduled_orders['Start Time'])
                    scheduled_orders['Finish Time'] = pd.to_datetime(scheduled_orders['Finish Time'])
                    scheduled_orders['Due Date'] = pd.to_datetime(scheduled_orders['Due Date'])
                    
                    makespan = (scheduled_orders['Finish Time'].max() -
                              scheduled_orders['Start Time'].min()).total_seconds() / 3600
                    
                    delayed_jobs = scheduled_orders[scheduled_orders['Finish Time'] > scheduled_orders['Due Date']]
                    if len(delayed_jobs) > 0:
                        avg_delay = (delayed_jobs['Finish Time'] - delayed_jobs['Due Date']).mean().total_seconds() / 3600
                    else:
                        avg_delay = 0
                    
                    st.write("**Schedule Performance:**")
                    st.write(f"""
                    - Total makespan: {makespan:.1f} hours
                    - Delayed jobs: {len(delayed_jobs)}
                    - Average delay: {avg_delay:.1f} hours
                    """)
                
                # Display schedule results first
                st.subheader("Schedule Results")
                display_cols = ['Job Number', 'Part Number', 'Due Date',
                              'JobPriority', 'operation sequence', 'Quantity', 'WorkCenter',
                              'Run Time', 'Setup Time', 'Place', 'Customer', 'Machine',
                              'Start Time', 'Finish Time']
                st.dataframe(scheduled_orders[display_cols].sort_values(['Start Time']), use_container_width=True)

                # Create tabs for visualizations
                st.subheader("Visualizations")
                tab1, tab2 = st.tabs(["Overview", "Gantt Chart"])
                
                with tab1:
                    # Create heatmap in the first tab
                    st.subheader("Work Center Utilization")
                    fig_heatmap = create_workload_heatmap(scheduled_orders, resources_df)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                with tab2:
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
                        total_duration = (job_ops['Finish Time'].max() - job_ops['Start Time'].min()).total_seconds() / 3600
                        
                        with job_col2:
                            st.write("**Job Details:**")
                            st.write(f"Part Number: {job_ops.iloc[0]['Part Number']}")
                            st.write(f"Number of Operations: {len(job_ops)}")
                            st.write(f"Total Processing Time: {job_ops['Run Time'].sum():.1f}h")
                            st.write(f"Total Setup Time: {job_ops['Setup Time'].sum():.1f}h")
                            st.write(f"Total Duration: {total_duration:.1f}h")
                    
                    try:
                        # Create Gantt chart with job highlighting
                        fig_gantt = create_gantt_chart(scheduled_orders, job_to_highlight)
                        st.plotly_chart(fig_gantt, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating Gantt chart: {str(e)}")
                
                # Convert datetime columns back to string for CSV export
                scheduled_orders['Start Time'] = scheduled_orders['Start Time'].dt.strftime('%Y-%m-%d %H:%M')
                scheduled_orders['Finish Time'] = scheduled_orders['Finish Time'].dt.strftime('%Y-%m-%d %H:%M')
                scheduled_orders['Due Date'] = scheduled_orders['Due Date'].dt.strftime('%Y-%m-%d')
                
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
