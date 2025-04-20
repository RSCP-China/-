import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Production Scheduler", layout="wide")

def load_production_orders(file):
    try:
        # Try different encodings, starting with GBK which is common for Chinese text
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
                df = pd.read_csv(file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error loading file with {encoding} encoding: {str(e)}")
                continue
        
        # Convert Due Date to datetime
        df['Due Date'] = pd.to_datetime(df['Due Date'], format='%d/%m/%Y')
        # Convert Run Time and Setup Time to float if they're not already
        df['Run Time'] = pd.to_numeric(df['Run Time'])
        df['Setup Time'] = pd.to_numeric(df['Setup Time'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_resources(file):
    try:
        # Try different encodings
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
                df = pd.read_csv(file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error loading file with {encoding} encoding: {str(e)}")
                continue
        return df
    except Exception as e:
        st.error(f"Error loading resources file: {str(e)}")
        return None

def validate_weights(weights):
    total = sum(weights.values())
    return abs(total - 100) < 0.01  # Allow for small floating point differences

def get_optimization_weights():
    st.sidebar.header("Optimization Weights")
    st.sidebar.info("Allocate weights to different optimization strategies. The sum must equal 100%")
    
    weights = {
        'makespan': st.sidebar.number_input("Minimize Total Makespan (%)", 
                                          min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'due_date': st.sidebar.number_input("Prioritize Due Dates (%)", 
                                          min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'utilization': st.sidebar.number_input("Maximize Resource Utilization (%)", 
                                             min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'setup_time': st.sidebar.number_input("Minimize Setup Times (%)", 
                                            min_value=0.0, max_value=100.0, value=25.0, step=5.0)
    }
    
    total = sum(weights.values())
    st.sidebar.write(f"Total: {total}%")
    
    if not validate_weights(weights):
        st.sidebar.error("Weights must sum to 100%")
        return None
    
    return {k: v/100.0 for k, v in weights.items()}  # Normalize to 0-1 range

def get_next_work_day(dt, total_hours):
    """Calculate the end time considering work hours (8AM-5PM)"""
    current_time = dt
    remaining_hours = total_hours
    
    while remaining_hours > 0:
        # If current time is before 8AM, move to 8AM
        if current_time.hour < 8:
            current_time = current_time.replace(hour=8, minute=0, second=0, microsecond=0)
        # If current time is after 5PM, move to 8AM next day
        elif current_time.hour >= 17:
            current_time = (current_time + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
        
        # Calculate hours until end of current work day
        hours_until_end = 17 - current_time.hour
        
        if remaining_hours <= hours_until_end:
            # Task can be completed today
            current_time = current_time + timedelta(hours=remaining_hours)
            remaining_hours = 0
        else:
            # Task continues to next day
            remaining_hours -= hours_until_end
            current_time = (current_time + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    
    return current_time

def create_schedule(orders_df, resources_df, weights):
    # Sort orders by priority (lower number = higher priority) and due date
    orders_df = orders_df.sort_values(['Priority', 'Due Date'])
    
    # Group resources by WorkCenter and Place
    resources_dict = {}
    for _, resource in resources_df.iterrows():
        key = (resource['WorkCenter'], resource['Place'])
        resources_dict[key] = {
            'available_machines': resource['Available Quantity'],
            'shift_hours': resource['Shift hours']
        }
    
    # Initialize machine availability times
    machine_times = {}
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM
    
    # Initialize all machines with start time
    for (work_center, place), resource in resources_dict.items():
        for machine in range(resource['available_machines']):
            machine_times[(work_center, place, machine)] = start_time
    
    schedule = []
    
    # Process orders
    for _, order in orders_df.iterrows():
        work_center = order['WorkCenter']
        place = order['Place']
        key = (work_center, place)
        
        if key not in resources_dict:
            continue
        
        resource = resources_dict[key]
        total_time = order['Run Time'] + order['Setup Time']
        
        # Find earliest available machine for this work center
        earliest_time = datetime.max
        best_machine = None
        
        for machine in range(resource['available_machines']):
            machine_key = (work_center, place, machine)
            if machine_key in machine_times:
                available_time = machine_times[machine_key]
                
                if available_time < earliest_time:
                    earliest_time = available_time
                    best_machine = machine
        
        if best_machine is None:
            continue
        
        machine_key = (work_center, place, best_machine)
        start_time = machine_times[machine_key]
        
        # Calculate end time considering work hours
        end_time = get_next_work_day(start_time, total_time)
        
        # Update machine availability time
        machine_times[machine_key] = end_time
        
        schedule.append({
            'Job Number': order['Job Number'],
            'Part Number': order['Part Number'],
            'WorkCenter': work_center,
            'Place': place,
            'Priority': order['Priority'],
            'Quantity': order['Quantity'],
            'Start Date': start_time.strftime('%Y-%m-%d %H:%M'),
            'End Date': end_time.strftime('%Y-%m-%d %H:%M'),
            'Machine ID': best_machine + 1,
            'Due Date': order['Due Date'].strftime('%Y-%m-%d'),
            'Total Hours': total_time,
            'Setup Time': order['Setup Time'],
            'Run Time': order['Run Time']
        })
    
    schedule_df = pd.DataFrame(schedule)
    
    # Calculate schedule metrics
    if not schedule_df.empty:
        schedule_df['Start Date'] = pd.to_datetime(schedule_df['Start Date'])
        schedule_df['End Date'] = pd.to_datetime(schedule_df['End Date'])
        schedule_df['Due Date'] = pd.to_datetime(schedule_df['Due Date'])
        
        # Calculate metrics based on weights
        makespan = (schedule_df['End Date'].max() - schedule_df['Start Date'].min()).total_seconds() / 3600
        late_orders = schedule_df[schedule_df['End Date'] > schedule_df['Due Date']]
        total_lateness = sum((late_orders['End Date'] - late_orders['Due Date']).dt.total_seconds() / 3600)
        total_setup_time = schedule_df['Setup Time'].sum()
        
        # Add metrics to the sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Schedule Metrics")
        st.sidebar.write(f"Total Makespan: {makespan:.2f} hours")
        st.sidebar.write(f"Late Orders: {len(late_orders)}")
        st.sidebar.write(f"Total Lateness: {total_lateness:.2f} hours")
        st.sidebar.write(f"Total Setup Time: {total_setup_time:.2f} hours")
    
    return schedule_df

def main():
    st.title("Production Scheduler")
    
    # Get optimization weights
    weights = get_optimization_weights()
    
    col1, col2 = st.columns(2)
    
    with col1:
        orders_file = st.file_uploader("Upload Production Orders (CSV)", type=['csv'])
    
    with col2:
        resources_file = st.file_uploader("Upload Resources Data (CSV)", type=['csv'])
    
    if orders_file and resources_file and weights:
        orders_df = load_production_orders(orders_file)
        resources_df = load_resources(resources_file)
        
        if orders_df is not None and resources_df is not None:
            st.subheader("Production Orders")
            st.dataframe(orders_df)
            
            st.subheader("Resources")
            st.dataframe(resources_df)
            
            if st.button("Generate Schedule"):
                with st.spinner("Generating production schedule..."):
                    schedule_df = create_schedule(orders_df, resources_df, weights)
                
                if schedule_df is not None and not schedule_df.empty:
                    st.success("Production Schedule Generated!")
                    st.subheader("Production Schedule")
                    st.dataframe(schedule_df)
                    
                    # Analysis
                    st.subheader("Schedule Analysis")
                    
                    # Check for late orders
                    late_orders = schedule_df[schedule_df['End Date'] > schedule_df['Due Date']]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Late Orders")
                        if not late_orders.empty:
                            st.warning(f"Number of late orders: {len(late_orders)}")
                            st.dataframe(late_orders[['Job Number', 'Part Number', 'End Date', 
                                                    'Due Date', 'Total Hours']])
                        else:
                            st.success("No late orders!")
                    
                    with col2:
                        st.write("Work Center Utilization")
                        utilization = schedule_df.groupby(['WorkCenter', 'Place']).agg({
                            'Total Hours': 'sum',
                            'Machine ID': 'nunique'
                        }).round(2)
                        utilization['Avg Hours per Machine'] = (utilization['Total Hours'] / 
                                                              utilization['Machine ID']).round(2)
                        st.dataframe(utilization)
                    
                    # Download schedule
                    csv = schedule_df.to_csv(index=False, encoding='gbk')  # Use GBK encoding for output
                    st.download_button(
                        "Download Schedule",
                        csv,
                        "production_schedule.csv",
                        "text/csv",
                        key='download-csv'
                    )

if __name__ == "__main__":
    main()