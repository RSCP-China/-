# Production Scheduler

An intelligent production scheduling application that optimizes manufacturing schedules based on multiple weighted objectives.

## Features

- Multi-objective optimization with adjustable weights
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
2. Adjust optimization weights in the sidebar:
   - Minimize Total Makespan
   - Prioritize Due Dates
   - Maximize Resource Utilization
   - Minimize Setup Times
3. Click "Generate Schedule" to create the production schedule
4. Review the schedule and analysis
5. Download the optimized schedule as CSV

## Schedule Analysis Features

- Late order detection
- Work center utilization statistics
- Setup time analysis
- Total makespan calculation
- Machine assignment details

## Notes

- All times are handled in 24-hour format
- Jobs are automatically split across work days
- Schedule respects standard work hours (8 AM to 5 PM)
- Chinese character support using GBK encoding
- Schedule can be downloaded in CSV format with proper encoding

## Support

For issues and feature requests, please create an issue in the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.