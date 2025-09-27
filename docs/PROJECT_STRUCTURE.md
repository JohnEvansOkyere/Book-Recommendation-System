# Project Structure Documentation

## Directory Organization

### Core Components
- `books_recommender/` - Main recommendation system package
- `artifacts/` - Generated models, data, and serialized objects
- `config/` - Configuration files
- `datasets/` - Raw datasets

### Scripts and Utilities
- `scripts/evaluation/` - Model evaluation scripts
- `scripts/visualization/` - Data visualization scripts
- `scripts/utilities/` - General utility scripts

### Documentation and Reports
- `docs/` - Project documentation
- `reports/performance/` - Performance analysis reports
- `reports/evaluation/` - Model evaluation reports

### Images and Media
- `images/visualizations/` - Data visualization charts
- `images/performance/` - Performance metric charts
- `images/screenshots/` - Application screenshots

### Testing
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/performance/` - Performance tests

### Logs and Backup
- `logs/` - Application and training logs
- `backup/` - Model and data backups

## File Naming Conventions

### Scripts
- `*_evaluation.py` - Model evaluation scripts
- `*_visualization.py` - Visualization scripts
- `*_analysis.py` - Data analysis scripts

### Reports
- `*_performance_report_*.txt` - Performance analysis reports
- `*_evaluation_report_*.txt` - Model evaluation reports

### Images
- `rating_distribution.png` - Rating distribution charts
- `user_activity.png` - User activity analysis
- `book_popularity.png` - Book popularity analysis
- `performance_metrics.png` - Performance comparison charts

## Maintenance

### Regular Tasks
1. Move new visualization files to `images/visualizations/`
2. Move new reports to appropriate `reports/` subdirectories
3. Clean up temporary files in `temp/`
4. Archive old logs to `backup/`

### File Organization
- Use descriptive names for all files
- Group related files in appropriate directories
- Keep the root directory clean
- Document any new directories or files