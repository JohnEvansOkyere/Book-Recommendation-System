#!/usr/bin/env python3
"""
Project Organization Script
=========================

This script helps organize the Book Recommendation System project
by creating necessary folders and moving files to appropriate locations.

Author: AI Assistant
Date: 2024
"""

import os
import shutil
import glob
from pathlib import Path

class ProjectOrganizer:
    """
    Organizes the project structure and moves files to appropriate locations
    """
    
    def __init__(self, project_root="."):
        """Initialize the project organizer"""
        self.project_root = Path(project_root)
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            "images/performance",
            "images/visualizations", 
            "images/screenshots",
            "scripts/evaluation",
            "scripts/visualization",
            "scripts/utilities",
            "reports/performance",
            "reports/evaluation",
            "docs/technical",
            "docs/user_guide",
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "logs/application",
            "logs/training",
            "backup/models",
            "backup/data"
        ]
        
        print("📁 Creating directory structure...")
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {directory}")
    
    def organize_visualization_files(self):
        """Move visualization files to appropriate folders"""
        print("\n📊 Organizing visualization files...")
        
        # Move PNG files to visualizations folder
        png_files = list(self.project_root.glob("*.png"))
        for file in png_files:
            dest = self.project_root / "images" / "visualizations" / file.name
            shutil.move(str(file), str(dest))
            print(f"  ✓ Moved: {file.name} → images/visualizations/")
    
    def organize_scripts(self):
        """Move evaluation and utility scripts to appropriate folders"""
        print("\n🔧 Organizing scripts...")
        
        # Evaluation scripts
        eval_scripts = [
            "model_evaluation.py",
            "enhanced_model_evaluation.py", 
            "test_improved_models.py"
        ]
        
        for script in eval_scripts:
            if (self.project_root / script).exists():
                dest = self.project_root / "scripts" / "evaluation" / script
                shutil.move(str(self.project_root / script), str(dest))
                print(f"  ✓ Moved: {script} → scripts/evaluation/")
        
        # Visualization scripts
        viz_scripts = [
            "visualization_analysis.py"
        ]
        
        for script in viz_scripts:
            if (self.project_root / script).exists():
                dest = self.project_root / "scripts" / "visualization" / script
                shutil.move(str(self.project_root / script), str(dest))
                print(f"  ✓ Moved: {script} → scripts/visualization/")
    
    def organize_reports(self):
        """Move report files to reports folder"""
        print("\n📋 Organizing reports...")
        
        # Performance reports
        report_files = list(self.project_root.glob("*performance_report*.txt"))
        for file in report_files:
            dest = self.project_root / "reports" / "performance" / file.name
            shutil.move(str(file), str(dest))
            print(f"  ✓ Moved: {file.name} → reports/performance/")
        
        # Evaluation reports
        eval_reports = list(self.project_root.glob("*evaluation_report*.txt"))
        for file in eval_reports:
            dest = self.project_root / "reports" / "evaluation" / file.name
            shutil.move(str(file), str(dest))
            print(f"  ✓ Moved: {file.name} → reports/evaluation/")
    
    def create_gitignore(self):
        """Create or update .gitignore file"""
        print("\n📝 Creating .gitignore...")
        
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.pkl
*.pickle
*.h5
*.hdf5

# Model files
artifacts/trained_model/*.pkl
artifacts/serialized_objects/*.pkl
artifacts/dataset/transformed_data/*.pkl

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Project specific
backup/
temp/
*.tmp

# Images (keep structure but ignore large files)
images/screenshots/*.png
images/performance/*.png
!images/visualizations/*.png

# Reports (keep structure but ignore large files)
reports/performance/*.txt
reports/evaluation/*.txt
"""
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        
        print("  ✓ Created .gitignore file")
    
    def create_project_structure_doc(self):
        """Create a project structure documentation"""
        print("\n📚 Creating project structure documentation...")
        
        structure_doc = """# Project Structure Documentation

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
"""
        
        doc_path = self.project_root / "docs" / "PROJECT_STRUCTURE.md"
        with open(doc_path, 'w') as f:
            f.write(structure_doc.strip())
        
        print("  ✓ Created project structure documentation")
    
    def create_cleanup_script(self):
        """Create a cleanup script for maintaining project organization"""
        print("\n🧹 Creating cleanup script...")
        
        cleanup_script = """#!/usr/bin/env python3
\"\"\"
Project Cleanup Script
====================

This script helps maintain project organization by:
- Moving files to appropriate directories
- Cleaning up temporary files
- Archiving old logs and reports

Usage: python cleanup_project.py
\"\"\"

import os
import shutil
import glob
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_temp_files():
    \"\"\"Remove temporary files\"\"\"
    temp_patterns = ['*.tmp', '*.temp', '*~', '*.bak']
    for pattern in temp_patterns:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"Removed: {file}")

def archive_old_logs():
    \"\"\"Archive logs older than 30 days\"\"\"
    log_dir = Path('logs')
    if not log_dir.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=30)
    
    for log_file in log_dir.rglob('*.log'):
        if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
            archive_dir = Path('backup/logs')
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(log_file), str(archive_dir / log_file.name))
            print(f"Archived: {log_file}")

def organize_new_files():
    \"\"\"Organize any new files that might have been created\"\"\"
    # Move new PNG files to visualizations
    for png_file in glob.glob('*.png'):
        dest = Path('images/visualizations') / png_file
        Path('images/visualizations').mkdir(parents=True, exist_ok=True)
        shutil.move(png_file, str(dest))
        print(f"Moved: {png_file} → images/visualizations/")
    
    # Move new report files
    for report_file in glob.glob('*_report_*.txt'):
        dest = Path('reports/performance') / report_file
        Path('reports/performance').mkdir(parents=True, exist_ok=True)
        shutil.move(report_file, str(dest))
        print(f"Moved: {report_file} → reports/performance/")

if __name__ == "__main__":
    print("🧹 Starting project cleanup...")
    cleanup_temp_files()
    archive_old_logs()
    organize_new_files()
    print("✅ Cleanup completed!")
"""
        
        cleanup_path = self.project_root / "cleanup_project.py"
        with open(cleanup_path, 'w') as f:
            f.write(cleanup_script.strip())
        
        # Make it executable
        os.chmod(cleanup_path, 0o755)
        print("  ✓ Created cleanup script")
    
    def organize_project(self):
        """Main method to organize the entire project"""
        print("🚀 Starting project organization...")
        print("="*50)
        
        # Create directory structure
        self.setup_directories()
        
        # Organize files
        self.organize_visualization_files()
        self.organize_scripts()
        self.organize_reports()
        
        # Create documentation and utilities
        self.create_gitignore()
        self.create_project_structure_doc()
        self.create_cleanup_script()
        
        print("\n✅ Project organization completed!")
        print("\n📋 Summary:")
        print("  • Created organized directory structure")
        print("  • Moved files to appropriate locations")
        print("  • Created .gitignore file")
        print("  • Created project structure documentation")
        print("  • Created cleanup script for maintenance")
        
        print("\n🔧 Next steps:")
        print("  • Review the organized structure")
        print("  • Update any hardcoded paths in your code")
        print("  • Run the cleanup script regularly: python cleanup_project.py")
        print("  • Keep the project organized as you add new files")


def main():
    """Main function to organize the project"""
    organizer = ProjectOrganizer()
    organizer.organize_project()


if __name__ == "__main__":
    main()
