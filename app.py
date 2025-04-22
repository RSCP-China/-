import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Translation dictionary
TRANSLATIONS = {
    'en': {
        'title': 'Production Scheduler',
        'optimization_weights': 'Optimization Weights',
        'date_range': 'Date Range',
        'start_date': 'Start Date',
        'end_date': 'End Date',
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
        'hours': 'hours',
        'visualization_tab': 'Visualization',
        'heatmap_title': 'Work Center Load Heat Map',
        'select_order': 'Select Order to Highlight',
        'load_level': 'Load Level',
        'time_span': 'Time Span',
        'work_centers': 'Work Centers',
        'load_percentage': 'Load Percentage',
        'high_load': 'High Load (>80%)',
        'medium_load': 'Medium Load (50-80%)',
        'low_load': 'Low Load (<50%)',
        'schedule_tab': 'Schedule',
        'analysis_tab': 'Analysis'
    },
    'zh': {
        'title': '生产排程系统',
        'optimization_weights': '优化权重',
        'date_range': '日期范围',
        'start_date': '开始日期',
        'end_date': '结束日期',
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
        'hours': '小时',
        'visualization_tab': '可视化',
        'heatmap_title': '工作中心负荷热图',
        'select_order': '选择订单以突出显示',
        'load_level': '负荷水平',
        'time_span': '时间范围',
        'work_centers': '工作中心',
        'load_percentage': '负荷百分比',
        'high_load': '高负荷 (>80%)',
        'medium_load': '中等负荷 (50-80%)',
        'low_load': '低负荷 (<50%)',
        'schedule_tab': '排程',
        'analysis_tab': '分析'
    }
}

def init_session_state():
    if 'language' not in st.session_state:
        st.session_state.language = None
    if 'selected_order' not in st.session_state:
        st.session_state.selected_order = 'None'
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    if 'schedule_df' not in st.session_state:
        st.session_state.schedule_df = None
    if 'resources_df' not in st.session_state:
        st.session_state.resources_df = None

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

def create_interactive_heatmap(schedule_df, resources_df, selected_order=None):
    """Create an interactive heat map showing work center loads and highlighting selected orders"""
    # Process dates and work centers
    schedule_df['Start Date'] = pd.to_datetime(schedule_df['Start Date'])
    schedule_df['End Date'] = pd.to_datetime(schedule_df['End Date'])
    date_range = pd.date_range(
        schedule_df['Start Date'].min(),
        schedule_df['End Date'].max(),
        freq='D'
    )
    work_centers = schedule_df['WorkCenter'].unique()
    
    # Initialize matrices
    load_data = np.zeros((len(work_centers), len(date_range)))
    order_highlight = np.zeros((len(work_centers), len(date_range)))
    text_data = [[[] for _ in range(len(date_range))] for _ in range(len(work_centers))]
    
    # Calculate loads and prepare highlighting
    for i, work_center in enumerate(work_centers):
        wc_schedule = schedule_df[schedule_df['WorkCenter'] == work_center]
        max_capacity = resources_df[
            resources_df['WorkCenter'] == work_center
        ]['Available Quantity'].iloc[0] * 24  # 24 hours per day
        
        for j, date in enumerate(date_range):
            date_schedule = wc_schedule[
                (wc_schedule['Start Date'].dt.date <= date.date()) &
                (wc_schedule['End Date'].dt.date >= date.date())
            ]
            
            # Calculate load
            daily_load = date_schedule['Total Hours'].sum()
            load_percentage = (daily_load / max_capacity) * 100
            load_data[i, j] = load_percentage
            
            # Store orders for this cell
            orders = date_schedule['Job Number'].tolist()
            text_data[i][j] = orders
            
            # Highlight selected order
            if selected_order and selected_order in orders:
                order_highlight[i, j] = 1
    
    return load_data, order_highlight, text_data, date_range, work_centers

def get_load_colorscale():
    """Get color scale for load percentage"""
    return [
        [0, 'rgb(0,0,255)'],      # Blue for <50%
        [0.5, 'rgb(0,255,0)'],    # Green for 50-80%
        [0.8, 'rgb(255,0,0)']     # Red for >80%
    ]

def get_highlight_colorscale():
    """Get color scale for order highlighting"""
    return [
        [0, 'rgba(128,128,128,0.3)'],  # Grey for non-highlighted
        [1, 'rgb(255,0,0)']            # Red for highlighted
    ]

def show_order_timeline(schedule_df):
    """Display a timeline view for a selected order"""
    st.subheader("Order Timeline View")
    
    # Order selection
    orders = sorted(schedule_df['Job Number'].unique())
    selected_order = st.selectbox(
        get_text('select_order'),
        ['None'] + list(orders)
    )
    
    if selected_order != 'None':
        # Convert datetime columns if they're strings
        if isinstance(schedule_df['Start Date'].iloc[0], str):
            schedule_df['Start Date'] = pd.to_datetime(schedule_df['Start Date'])
            schedule_df['End Date'] = pd.to_datetime(schedule_df['End Date'])
            schedule_df['Due Date'] = pd.to_datetime(schedule_df['Due Date'])
        
        # Get the selected order data
        order_data = schedule_df[schedule_df['Job Number'] == selected_order].copy()
        order_data['Duration'] = (order_data['End Date'] - order_data['Start Date']).dt.total_seconds() / 3600
        
        # Show order details
        st.markdown(f"""
        **Order Details:**
        - Job Number: {selected_order}
        - Part Number: {order_data['Part Number'].iloc[0]}
        - Due Date: {order_data['Due Date'].iloc[0].strftime('%Y-%m-%d')}
        - Total Hours: {order_data['Total Hours'].sum():.1f}
        """)
        
        # Create timeline visualization
        fig = go.Figure()
        
        # Add bars for each work step
        for idx, row in order_data.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Duration']],
                y=[f"{row['WorkCenter']} - {row['Place']}"],
                orientation='h',
                name=f"Machine {row['Machine ID']}",
                text=[f"Duration: {row['Duration']:.1f}h<br>Setup: {row['Setup Time']}h<br>Run: {row['Run Time']}h<br>"
                      f"Start: {row['Start Date'].strftime('%Y-%m-%d %H:%M')}<br>"
                      f"End: {row['End Date'].strftime('%Y-%m-%d %H:%M')}"],
                hoverinfo='text',
                marker=dict(color='rgb(55, 83, 109)')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Timeline for Order {selected_order}",
            xaxis_title="Duration (hours)",
            yaxis_title="Work Centers",
            showlegend=True,
            height=400,
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed schedule
        st.subheader("Detailed Schedule")
        display_df = order_data[[
            'WorkCenter', 'Place', 'Machine ID',
            'Start Date', 'End Date', 'Setup Time', 'Run Time', 'Total Hours'
        ]].sort_values('Start Date').copy()
        
        # Format datetime columns for display
        display_df['Start Date'] = display_df['Start Date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['End Date'] = display_df['End Date'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df)
    else:
        st.info("Select an order to view its timeline.")


def show_heatmap_tab(schedule_df, resources_df):
    """Display the base heat map without order highlighting"""
    st.subheader(get_text('heatmap_title'))
    
    # Get overall date range
    min_date = schedule_df['Start Date'].dt.date.min()
    max_date = schedule_df['End Date'].dt.date.max()
    
    # Date range selector
    st.write(get_text('date_range'))
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(get_text('start_date'), min_date)
    with col2:
        end_date = st.date_input(get_text('end_date'), max_date)
    
    # Filter schedule based on selected date range
    filtered_schedule = schedule_df[
        (schedule_df['Start Date'].dt.date <= end_date) &
        (schedule_df['End Date'].dt.date >= start_date)
    ]
    
    # Create filtered heatmap data
    load_data, _, text_data, dates, centers = create_interactive_heatmap(
        filtered_schedule,
        resources_df,
        None  # No order highlighting in base heat map
    )
    
    # Create figure with load heat map
    fig = go.Figure()
    if len(dates) > 0:  # Only add heatmap if we have data
        fig.add_trace(go.Heatmap(
            z=load_data,
            x=[d.strftime('%Y-%m-%d') for d in dates],
            y=centers,
            text=text_data,
            hoverongaps=False,
            colorscale=get_load_colorscale(),
            showscale=True,
            colorbar=dict(
                title=get_text('load_percentage'),
                ticktext=[
                    get_text('low_load'),
                    get_text('medium_load'),
                    get_text('high_load')
                ],
                tickvals=[25, 65, 90]
            )
        ))
    
    # Update layout with better spacing and controls
    fig.update_layout(
        xaxis_title=get_text('time_span'),
        yaxis_title=get_text('work_centers'),
        height=600,
        margin=dict(t=30, b=50, l=100, r=50),
        xaxis=dict(
            tickangle=-45,
            tickformat='%Y-%m-%d',
            tickmode='auto',
            nticks=20
        )
    )
    
    if filtered_schedule.empty:
        st.warning("No data available for the selected date range.")
    else:
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        total_hours = filtered_schedule['Total Hours'].sum()
        num_orders = filtered_schedule['Job Number'].nunique()
        st.markdown(f"""
        **Selected Period Summary:**
        - Total Work Hours: {total_hours:.1f}
        - Number of Orders: {num_orders}
        """)

def show_order_highlight_tab(schedule_df):
    """Display a timeline view for a selected order"""
    show_order_timeline(schedule_df)

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
            
            # Store data in session state for persistence
            if 'schedule_df' not in st.session_state:
                generate_button = st.button(get_text('generate_schedule'))
                if generate_button:
                    with st.spinner("..."):
                        st.session_state.schedule_df = create_schedule(orders_df, resources_df, weights)
                        st.session_state.resources_df = resources_df
            else:
                if st.button(get_text('generate_schedule')):
                    with st.spinner("..."):
                        st.session_state.schedule_df = create_schedule(orders_df, resources_df, weights)
                        st.session_state.resources_df = resources_df

            # Show results if we have a schedule
            if 'schedule_df' in st.session_state and st.session_state.schedule_df is not None and not st.session_state.schedule_df.empty:
                    st.success(get_text('schedule_generated'))
                    
                    # Create tabs
                    tabs = st.tabs([
                        get_text('schedule_tab'),
                        get_text('visualization_tab'),
                        'Order Highlighting',
                        get_text('analysis_tab')
                    ])
                    
                    # Schedule tab
                    with tabs[0]:
                        st.subheader(get_text('title'))
                        st.dataframe(st.session_state.schedule_df)
                        
                        # Download schedule
                        csv = st.session_state.schedule_df.to_csv(index=False, encoding='gbk')
                        st.download_button(
                            get_text('download_schedule'),
                            csv,
                            "production_schedule.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    
                    # Visualization tab (base heat map)
                    with tabs[1]:
                        show_heatmap_tab(st.session_state.schedule_df, st.session_state.resources_df)
                    
                    # Order Highlighting tab
                    with tabs[2]:
                        show_order_highlight_tab(st.session_state.schedule_df)
                        
                    # Analysis tab
                    with tabs[3]:
                        # Check for late orders
                        late_orders = st.session_state.schedule_df[
                            st.session_state.schedule_df['End Date'] > st.session_state.schedule_df['Due Date']
                        ]
                        
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
                            utilization = st.session_state.schedule_df.groupby(['WorkCenter', 'Place']).agg({
                                'Total Hours': 'sum',
                                'Machine ID': 'nunique'
                            }).round(2)
                            utilization['Avg Hours per Machine'] = (utilization['Total Hours'] /
                                                                  utilization['Machine ID']).round(2)
                            st.dataframe(utilization)

if __name__ == "__main__":
    main()