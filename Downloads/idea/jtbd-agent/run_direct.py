#!/usr/bin/env python
"""
Direct JTBD Agent Runner - Runs DSPy pipeline without Prefect
"""

import sys
import os
from pathlib import Path
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from contracts.idea_v1 import IdeaV1
from core.pipeline import run_pipeline
from telemetry.metrics import summarize

def run_analysis_direct(idea_file, output_dir=None, assets_dir=None):
    """Run JTBD analysis directly without Prefect"""
    
    # Load and validate the idea FIRST (before creating any directories)
    with open(idea_file, 'r') as f:
        idea_data = json.load(f)
    
    # Create IdeaV1 object - this will fail if validation fails
    idea = IdeaV1(**idea_data)
    
    # Only create directories AFTER successful validation
    from datetime import datetime
    
    # Extract idea name from filename
    idea_name = Path(idea_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_folder = datetime.now().strftime("%Y-%m-%d")
    
    # Default organized structure with subdirectories
    if output_dir is None:
        base_dir = f"reports/{date_folder}/{idea_name}_{timestamp}"
        output_dir = base_dir
        
        # Create organized subdirectories for different output types
        subdirs = {
            "gamma": f"{base_dir}/gamma",      # Gamma presentation files
            "csv": f"{base_dir}/csv",          # CSV exports
            "json": f"{base_dir}/json",        # JSON data dumps
            "assets": f"{base_dir}/assets",    # Images, charts, etc.
            "logs": f"{base_dir}/logs",        # Execution logs
            "cache": f"{base_dir}/cache"       # Cached intermediate results
        }
        
        for subdir in subdirs.values():
            Path(subdir).mkdir(parents=True, exist_ok=True)
    else:
        # Still create subdirs even if custom output_dir provided
        subdirs = {
            "gamma": f"{output_dir}/gamma",
            "csv": f"{output_dir}/csv",
            "json": f"{output_dir}/json",
            "assets": f"{output_dir}/assets",
            "logs": f"{output_dir}/logs",
            "cache": f"{output_dir}/cache"
        }
        
        for subdir in subdirs.values():
            Path(subdir).mkdir(parents=True, exist_ok=True)
    
    if assets_dir is None:
        assets_dir = subdirs["assets"]
    
    # Create main output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(assets_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the JTBD report
    print(f"🚀 Running JTBD analysis on: {idea.title}")
    print(f"📊 Hunches: {idea.hunches}")
    print(f"🔑 Using OpenAI API Key: {'*' * 20 + os.getenv('OPENAI_API_KEY', '')[-4:]}")
    print(f"🤖 Running DSPy pipeline directly (no Prefect)...")
    print("-" * 50)
    
    try:
        # Run the pipeline directly
        result = run_pipeline(idea, assets_dir, output_dir)
        
        # Determine subdirectory paths
        if output_dir.endswith(("/gamma", "/csv", "/json", "/assets", "/logs", "/cache")):
            # Custom output dir, use parent
            gamma_dir = Path(output_dir).parent / "gamma"
            csv_dir = Path(output_dir).parent / "csv"
            json_dir = Path(output_dir).parent / "json"
        else:
            # Use subdirectories
            gamma_dir = Path(output_dir) / "gamma"
            csv_dir = Path(output_dir) / "csv"
            json_dir = Path(output_dir) / "json"
        
        # Ensure dirs exist
        gamma_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the Gamma markdown output in gamma subdirectory
        output_file = gamma_dir / "presentation.md"
        output_file.write_text(result["gamma_md"], encoding="utf-8")
        
        # Print summary
        print("\n" + "="*50)
        print("✅ ANALYSIS COMPLETE")
        print("="*50)
        print(summarize(result["metrics"]))
        print(f"\n📄 Gamma presentation saved to: {output_file}")
        
        # Also save as HTML for preview in gamma subdirectory
        html_file = gamma_dir / "presentation.html"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>JTBD Analysis - {idea.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        .score {{ font-size: 2em; font-weight: bold; color: #27ae60; }}
        .delta {{ font-size: 1.5em; color: #e74c3c; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>JTBD Analysis Report</h1>
    <h2>{idea.title}</h2>
    <div style="white-space: pre-wrap;">{result.get("gamma_md", "No content generated")}</div>
</body>
</html>"""
        html_file.write_text(html_content)
        print(f"🌐 HTML preview saved to: {html_file}")
        
        # Export to CSV if we have the data
        try:
            from core.export_spreadsheet import export_to_csv
            
            # Save JSON data dumps for debugging/analysis
            json_data = {
                "idea": idea.model_dump(),
                "metrics": result.get("metrics", {}),
                "s0": result.get("s0").model_dump() if result.get("s0") else None,
                "s1": result.get("s1").model_dump() if result.get("s1") else None,
                "delta": result.get("delta", 0),
                "akr": result.get("akr", 0),
                "assumptions": [a.model_dump() for a in result.get("assumptions", [])] if result.get("assumptions") else [],
                "jobs": [j.model_dump() for j in result.get("jobs", [])] if result.get("jobs") else [],
                "plan": result.get("plan").model_dump() if result.get("plan") else None
            }
            
            json_file = json_dir / "analysis_data.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"📋 JSON data saved to: {json_file}")
            
            # Export CSVs if we have the necessary data
            if all(key in result for key in ["s0", "s1", "delta", "akr", "assumptions", "jobs", "plan"]):
                csv_files = export_to_csv(
                    str(csv_dir),  # Save in csv subdirectory
                    result["s0"],
                    result["s1"],
                    result["delta"],
                    result["akr"],
                    result["assumptions"],
                    result["jobs"],
                    result["plan"]
                )
                print(f"📊 CSV exports saved to: {csv_dir}/")
            else:
                print("⚠️ Some data missing, skipping CSV export")
                
        except Exception as e:
            print(f"⚠️ Could not export to CSV: {e}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Run JTBD Agent Analysis (Direct)')
    parser.add_argument('idea_file', help='Path to idea JSON file')
    parser.add_argument('--output', default=None, help='Output directory (auto-organized by date if not specified)')
    parser.add_argument('--assets', default=None, help='Assets directory')
    
    args = parser.parse_args()
    
    try:
        run_analysis_direct(args.idea_file, args.output, args.assets)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()