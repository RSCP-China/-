# Production Scheduler

An intelligent production scheduling application that optimizes manufacturing schedules based on multiple weighted objectives.

## Features

- Multi-objective optimization with adjustable weights
- Intelligent order batching for efficiency
- Dynamic date range filtering for visualization
- Standard work hours (8 AM to 5 PM) scheduling
- Support for multiple work centers and locations
- Real-time schedule analysis
- Chinese language support
- CSV import/export functionality

## Local Setup

1. Install Python 3.8 or newer
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud

1. Create a GitHub Repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. Deploy on Streamlit Cloud:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (main), and file (app.py)
   - Click "Deploy"

## Input File Formats

### Production Orders (CSV)
Required columns:
- Job Number
- Part Number
- Due Date (DD/MM/YYYY)
- Priority
- Quantity
- WorkCenter
- Run Time (hours)
- Setup Time (hours)
- Place
- Customer

### Resources (CSV)
Required columns:
- WorkCenter
- Available Quantity
- Shift hours
- Shift Pattern
- Place

## Using the Application

1. Upload Production Orders and Resources CSV files

2. Configure Optimization Settings in the sidebar:
   - Adjust optimization weights:
     * Minimize Total Makespan
     * Prioritize Due Dates
     * Maximize Resource Utilization
     * Minimize Setup Times
   - Set batching parameters:
     * Specify time window (in days) for batching similar orders
     * Orders with same part number within the window will be combined

3. Click "Generate Schedule" to create the production schedule

4. Review schedule and analysis across different tabs:
   - Schedule Tab:
     * View complete schedule with batched orders
     * Expand batched orders (prefixed with "BATCH_") to see details
     * See original orders, total quantities, and combined times
   - Visualization Tab:
     * Select date range to filter the heat map view
     * Analyze work center loads for specific periods
     * View summary statistics for the selected timeframe
   - Analysis Tab:
     * Check late orders and utilization statistics
     * Review setup time analysis and makespan

5. Download the optimized schedule as CSV

## Schedule Analysis Features

- Late order detection
- Work center utilization statistics
- Setup time analysis
- Total makespan calculation
- Machine assignment details
- Batch order tracking and analysis
- Date-range filtered load analysis

## Advanced Features

### Order Batching
- Automatically combines orders with the same part number
- Configurable time window for batching flexibility
- Maintains traceability with original order information
- Optimizes setup times and machine utilization
- Preserves highest priority and earliest due date

### Heat Map Visualization
- Interactive date range selection
- Work center load visualization
- Color-coded capacity utilization
- Detailed order information on hover
- Period-specific summary statistics

## Notes

- All times are handled in 24-hour format
- Jobs are automatically split across work days
- Schedule respects standard work hours (8 AM to 5 PM)
- Chinese character support using GBK encoding
- Schedule can be downloaded in CSV format with proper encoding
- Batched orders maintain full traceability to original orders
- Heat map visualization can be filtered by date range

## Support

For issues and feature requests, please create an issue in the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.