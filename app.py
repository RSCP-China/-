import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Translation dictionary
TRANSLATIONS = {
    'en': {
        'title': 'Production Scheduler',
        'optimization_weights': 'Optimization Weights',
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
        'schedule_generated': 'Production Schedule Generated!',
        'schedule_analysis': 'Schedule Analysis',
        'late_orders': 'Late Orders',
        'no_late_orders': 'No late orders!',
        'work_center_util': 'Work Center Utilization',
        'download_schedule': 'Download Schedule',
        'num_late_orders': 'Number of late orders',
        'metrics': 'Schedule Metrics',
        'total_makespan': 'Total Makespan',
        'total_lateness': 'Total Lateness',
        'total_setup': 'Total Setup Time',
        'hours': 'hours'
    },
    'zh': {
        'title': '生产排程系统',
        'optimization_weights': '优化权重',
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
        'schedule_generated': '排程已生成！',
        'schedule_analysis': '排程分析',
        'late_orders': '延期订单',
        'no_late_orders': '没有延期订单！',
        'work_center_util': '工作中心利用率',
        'download_schedule': '下载排程',
        'num_late_orders': '延期订单数量',
        'metrics': '排程指标',
        'total_makespan': '总生产时间',
        'total_lateness': '总延期时间',
        'total_setup': '总设置时间',
        'hours': '小时'
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
    
    return {k: v/100.0 for k, v in weights.items()}

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
        st.sidebar.subheader(get_text('metrics'))
        st.sidebar.write(f"{get_text('total_makespan')}: {makespan:.2f} {get_text('hours')}")
        st.sidebar.write(f"{get_text('late_orders')}: {len(late_orders)}")
        st.sidebar.write(f"{get_text('total_lateness')}: {total_lateness:.2f} {get_text('hours')}")
        st.sidebar.write(f"{get_text('total_setup')}: {total_setup_time:.2f} {get_text('hours')}")
    
    return schedule_df

def main():
    init_session_state()
    if not language_selector():
        return

    st.title(get_text('title'))
    
    # Get optimization weights
    weights = get_optimization_weights()
    
    col1, col2 = st.columns(2)
    
    with col1:
        orders_file = st.file_uploader(get_text('upload_orders'), type=['csv'])
    
    with col2:
        resources_file = st.file_uploader(get_text('upload_resources'), type=['csv'])
    
    if orders_file and resources_file and weights:
        orders_df = load_production_orders(orders_file)
        resources_df = load_resources(resources_file)
        
        if orders_df is not None and resources_df is not None:
            st.subheader(get_text('upload_orders'))
            st.dataframe(orders_df)
            
            st.subheader(get_text('upload_resources'))
            st.dataframe(resources_df)
            
            if st.button(get_text('generate_schedule')):
                with st.spinner("..."):
                    schedule_df = create_schedule(orders_df, resources_df, weights)
                
                if schedule_df is not None and not schedule_df.empty:
                    st.success(get_text('schedule_generated'))
                    st.subheader(get_text('title'))
                    st.dataframe(schedule_df)
                    
                    # Analysis
                    st.subheader(get_text('schedule_analysis'))
                    
                    # Check for late orders
                    late_orders = schedule_df[schedule_df['End Date'] > schedule_df['Due Date']]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(get_text('late_orders'))
                        if not late_orders.empty:
                            st.warning(f"{get_text('num_late_orders')}: {len(late_orders)}")
                            st.dataframe(late_orders[['Job Number', 'Part Number', 'End Date',
                                                    'Due Date', 'Total Hours']])
                        else:
                            st.success(get_text('no_late_orders'))
                    
                    with col2:
                        st.write(get_text('work_center_util'))
                        utilization = schedule_df.groupby(['WorkCenter', 'Place']).agg({
                            'Total Hours': 'sum',
                            'Machine ID': 'nunique'
                        }).round(2)
                        utilization['Avg Hours per Machine'] = (utilization['Total Hours'] /
                                                              utilization['Machine ID']).round(2)
                        st.dataframe(utilization)
                    
                    # Download schedule
                    csv = schedule_df.to_csv(index=False, encoding='gbk')
                    st.download_button(
                        get_text('download_schedule'),
                        csv,
                        "production_schedule.csv",
                        "text/csv",
                        key='download-csv'
                    )

if __name__ == "__main__":
    main()