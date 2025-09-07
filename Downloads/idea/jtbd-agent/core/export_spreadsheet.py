"""
Spreadsheet export for JTBD analysis results
Exports to CSV and optionally Excel formats
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
from contracts.scorecard_v1 import ScorecardV1
from contracts.assumption_v1 import AssumptionV1
from contracts.job_v1 import JobV1
from contracts.validation_plan_v1 import ValidationPlanV1

def export_to_csv(
    output_dir: str,
    s0: ScorecardV1,
    s1: ScorecardV1, 
    delta: float,
    akr: float,
    assumptions: List[AssumptionV1],
    jobs: List[JobV1],
    plan: ValidationPlanV1
) -> Dict[str, str]:
    """Export JTBD analysis to multiple CSV files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    files_created = {}
    
    # 1. Scores Summary
    scores_file = output_path / "scores_summary.csv"
    with open(scores_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Initial', 'Final', 'Change'])
        writer.writerow(['Total Score', s0.total, s1.total, f"+{delta:.2f}"])
        writer.writerow(['AKR (Risk)', '', '', f"{akr:.2f}"])
        writer.writerow([])  # blank line
        writer.writerow(['Criterion', 'Initial Score', 'Final Score', 'Change', 'Final Rationale'])
        
        for c0, c1 in zip(s0.criteria, s1.criteria):
            change = c1.score - c0.score
            writer.writerow([c0.name, c0.score, c1.score, f"{change:+.1f}", c1.rationale])
    
    files_created['scores'] = str(scores_file)
    
    # 2. Assumptions Tracker
    assumptions_file = output_path / "assumptions_tracker.csv"
    with open(assumptions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Level', 'Confidence', 'Risk Score', 'Assumption Text', 'Evidence', 'Validation Status'])
        
        for a in sorted(assumptions, key=lambda x: (x.level, -x.confidence)):
            risk = (1 - a.confidence) * (4 - a.level) / 3  # Higher risk for low confidence + high level
            evidence_str = '; '.join(a.evidence) if a.evidence else 'None'
            writer.writerow([
                a.assumption_id,
                f"L{a.level}",
                f"{a.confidence:.0%}",
                f"{risk:.2f}",
                a.text,
                evidence_str,
                'Pending'  # Status column for tracking
            ])
    
    files_created['assumptions'] = str(assumptions_file)
    
    # 3. JTBD Analysis
    jtbd_file = output_path / "jtbd_analysis.csv"
    with open(jtbd_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Job ID', 'Job Statement', 'Push Forces', 'Pull Forces', 'Anxiety Forces', 'Inertia Forces'])
        
        for job in jobs:
            forces = job.forces
            writer.writerow([
                job.job_id,
                job.statement,
                '; '.join(forces.get('push', [])),
                '; '.join(forces.get('pull', [])),
                '; '.join(forces.get('anxiety', [])),
                '; '.join(forces.get('inertia', []))
            ])
    
    files_created['jtbd'] = str(jtbd_file)
    
    # 4. Validation Plan
    validation_file = output_path / "validation_plan.csv"
    with open(validation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Budget', 'Experiment ID', 'Hypothesis', 'Design', 'Metric', 'Success Criteria', 'EVI'])
        
        for stage in plan.stages:
            for exp in stage.experiments:
                writer.writerow([
                    stage.stage,
                    f"${stage.budget:.0f}",
                    exp.exp_id,
                    exp.hypothesis,
                    exp.design,
                    exp.metric,
                    exp.success,
                    f"{exp.evi:.2f}"
                ])
    
    files_created['validation'] = str(validation_file)
    
    # 5. Master Tracker (combines key data)
    master_file = output_path / "jtbd_master_tracker.csv"
    with open(master_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Summary section
        writer.writerow(['JTBD Analysis Master Tracker'])
        writer.writerow([])
        writer.writerow(['Summary Metrics'])
        writer.writerow(['Initial Score', 'Final Score', 'Delta', 'AKR'])
        writer.writerow([s0.total, s1.total, f"+{delta:.2f}", akr])
        writer.writerow([])
        
        # Quick reference
        writer.writerow(['Key Insights'])
        writer.writerow(['Total Assumptions', len(assumptions)])
        writer.writerow(['High Risk (L3)', sum(1 for a in assumptions if a.level == 3)])
        writer.writerow(['Total JTBD Perspectives', len(jobs)])
        writer.writerow(['Validation Stages', len(plan.stages)])
        total_experiments = sum(len(stage.experiments) for stage in plan.stages)
        writer.writerow(['Total Experiments', total_experiments])
        total_budget = sum(stage.budget for stage in plan.stages)
        writer.writerow(['Total Validation Budget', f"${total_budget:.0f}"])
        
    files_created['master'] = str(master_file)
    
    print(f"\n📊 Spreadsheets exported to {output_path}/")
    for name, path in files_created.items():
        print(f"  - {Path(path).name}")
    
    return files_created

def export_comparison_matrix(
    output_dir: str,
    ideas_results: List[Dict[str, Any]]
) -> str:
    """Export comparison matrix for multiple ideas"""
    
    output_path = Path(output_dir)
    comparison_file = output_path / "ideas_comparison.csv"
    
    with open(comparison_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        headers = ['Idea Title', 'Initial Score', 'Final Score', 'Delta', 'AKR', 
                   'Assumptions Count', 'High Risk Count', 'Top JTBD', 'Validation Budget']
        writer.writerow(headers)
        
        # Data rows
        for result in ideas_results:
            writer.writerow([
                result['title'],
                result['s0'].total,
                result['s1'].total,
                result['delta'],
                result['akr'],
                len(result['assumptions']),
                sum(1 for a in result['assumptions'] if a.level == 3),
                result['jobs'][0].statement if result['jobs'] else 'N/A',
                sum(e.budget_usd for e in result['plan'].experiments)
            ])
    
    return str(comparison_file)