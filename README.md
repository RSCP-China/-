# Production Scheduling Optimizer

An advanced production scheduling tool that uses constraint programming to optimize manufacturing schedules based on multiple weighted objectives.

## Features

- Multi-objective optimization with adjustable weights for:
  - Minimizing total makespan
  - Prioritizing due dates
  - Maximizing resource utilization
  - Minimizing setup times
- Support for multiple work centers and locations
- Handles parallel machine scheduling
- Considers setup times and run times
- Real-time schedule analysis and visualization
- Export schedules to CSV

## Installation

1. Ensure you have Python 3.8 or newer installed
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Input File Formats

### Production Orders (CSV)
Required columns:
- Job Number: Order identifier
- Part Number: Product identifier
- Due Date: Format DD/MM/YYYY
- Priority: Numerical priority (lower number = higher priority)
- Quantity: Number of units
- WorkCenter: Work center name
- Run Time: Processing time in hours
- Setup Time: Setup time in hours
- Place: Location identifier
- Customer: Customer identifier

### Resources (CSV)
Required columns:
- WorkCenter: Work center name
- Available Quantity: Number of parallel machines
- Shift hours: Working hours per shift
- Shift Pattern: Shift pattern identifier
- Place: Location identifier

## Running the Application

1. Navigate to the application directory:
   ```bash
   cd path/to/production_scheduling
   ```

2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Access the application in your web browser (typically http://localhost:8501)

## Using the Optimizer

1. Upload your Production Orders and Resources CSV files
2. Adjust optimization weights in the sidebar:
   - Allocate percentages to each optimization strategy
   - Ensure weights sum to 100%
3. Click "Generate Optimized Schedule" to run the optimization
4. Review the schedule and analysis
5. Download the optimized schedule as CSV

## Schedule Analysis

The application provides:
- Late order identification
- Work center utilization statistics
- Machine assignment details
- Total processing times
- Setup time analysis

## Output Format

The generated schedule includes:
- Job and part information
- Start and end dates/times
- Assigned work center and machine
- Processing duration
- Setup and run times
- Due date compliance status

## Notes

- The optimizer uses Google OR-Tools CP-SAT solver
- Maximum solving time is set to 60 seconds
- Schedules consider current date/time as the starting point
- All times are converted to hours internally for optimization