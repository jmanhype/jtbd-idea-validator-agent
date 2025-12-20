import React, { useMemo } from 'react';

interface ResultsViewProps {
  content: string;
  path: string;
  onBack: () => void;
}

interface ParsedResult {
  deltaScore: string;
  akr: string;
  initialScores: Array<{ name: string; score: number; rationale: string }>;
  finalScores: Array<{ name: string; score: number; rationale: string }>;
  assumptions: string[];
  jobs: string[];
  validation: string[];
}

function parseMarkdown(content: string): ParsedResult {
  const result: ParsedResult = {
    deltaScore: '0',
    akr: '0',
    initialScores: [],
    finalScores: [],
    assumptions: [],
    jobs: [],
    validation: [],
  };

  // Parse delta and AKR
  const headerMatch = content.match(/\*\*ΔScore:\*\*\s*([+-]?\d+\.?\d*)\s*·\s*\*\*AKR:\*\*\s*(\d+\.?\d*)/);
  if (headerMatch) {
    result.deltaScore = headerMatch[1];
    result.akr = headerMatch[2];
  }

  // Parse initial scores
  const initialSection = content.match(/## Scores \(Initial\)([\s\S]*?)(?=## Scores \(Final\)|$)/);
  if (initialSection) {
    const rows = initialSection[1].match(/\| ([^|]+) \| (\d+\.?\d*) \| ([^|]+) \|/g);
    if (rows) {
      result.initialScores = rows.slice(1).map((row) => {
        const match = row.match(/\| ([^|]+) \| (\d+\.?\d*) \| ([^|]+) \|/);
        return {
          name: match?.[1]?.trim() || '',
          score: parseFloat(match?.[2] || '0'),
          rationale: match?.[3]?.trim() || '',
        };
      }).filter((s) => s.name && !s.name.includes('---'));
    }
  }

  // Parse final scores
  const finalSection = content.match(/## Scores \(Final\)([\s\S]*?)(?=!\[|## Assumptions|$)/);
  if (finalSection) {
    const rows = finalSection[1].match(/\| ([^|]+) \| (\d+\.?\d*) \| ([^|]+) \|/g);
    if (rows) {
      result.finalScores = rows.slice(1).map((row) => {
        const match = row.match(/\| ([^|]+) \| (\d+\.?\d*) \| ([^|]+) \|/);
        return {
          name: match?.[1]?.trim() || '',
          score: parseFloat(match?.[2] || '0'),
          rationale: match?.[3]?.trim() || '',
        };
      }).filter((s) => s.name && !s.name.includes('---'));
    }
  }

  // Parse assumptions
  const assumptionsSection = content.match(/## Assumptions([\s\S]*?)(?=## JTBD|$)/);
  if (assumptionsSection) {
    const lines = assumptionsSection[1].match(/- L\d+ \([^)]+\) — .+/g);
    if (lines) {
      result.assumptions = lines.map((l) => l.replace(/^- /, ''));
    }
  }

  // Parse JTBD
  const jtbdSection = content.match(/## JTBD \(5 perspectives\)([\s\S]*?)(?=## Validation|$)/);
  if (jtbdSection) {
    const lines = jtbdSection[1].match(/- .+/g);
    if (lines) {
      result.jobs = lines.map((l) => l.replace(/^- /, ''));
    }
  }

  // Parse validation
  const validationSection = content.match(/## Validation Plan([\s\S]*?)(?=\*doc:\*|$)/);
  if (validationSection) {
    const lines = validationSection[1].match(/- \*\*.+/g);
    if (lines) {
      result.validation = lines.map((l) => l.replace(/^- /, ''));
    }
  }

  return result;
}

export function ResultsView({ content, path, onBack }: ResultsViewProps) {
  const parsed = useMemo(() => parseMarkdown(content), [content]);

  const getScoreColor = (score: number) => {
    if (score >= 8) return '#22c55e';
    if (score >= 6) return '#eab308';
    return '#ef4444';
  };

  return (
    <div className="results-view">
      <div className="results-header">
        <button onClick={onBack} className="btn-back">
          ← Back
        </button>
        <h2>Analysis Results</h2>
      </div>

      <div className="results-summary">
        <div className="summary-card delta">
          <span className="label">Score Change</span>
          <span className="value">{parsed.deltaScore}</span>
        </div>
        <div className="summary-card akr">
          <span className="label">Assumption Kill Rate</span>
          <span className="value">{parsed.akr}</span>
        </div>
      </div>

      <div className="results-grid">
        <section className="results-section scores">
          <h3>Final Scores</h3>
          <div className="score-bars">
            {parsed.finalScores.map((score) => (
              <div key={score.name} className="score-bar">
                <div className="score-info">
                  <span className="score-name">{score.name}</span>
                  <span className="score-value">{score.score}/10</span>
                </div>
                <div className="bar-bg">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${score.score * 10}%`,
                      backgroundColor: getScoreColor(score.score),
                    }}
                  />
                </div>
                <p className="score-rationale">{score.rationale.split('|')[0].replace('arb: ', '')}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="results-section assumptions">
          <h3>Key Assumptions</h3>
          <ul>
            {parsed.assumptions.map((assumption, i) => (
              <li key={i}>{assumption}</li>
            ))}
          </ul>
        </section>

        <section className="results-section jobs">
          <h3>Jobs to be Done</h3>
          <ul>
            {parsed.jobs.map((job, i) => (
              <li key={i}>{job}</li>
            ))}
          </ul>
        </section>

        <section className="results-section validation">
          <h3>Validation Plan</h3>
          <ul>
            {parsed.validation.map((item, i) => (
              <li key={i} dangerouslySetInnerHTML={{ __html: item.replace(/\*\*/g, '') }} />
            ))}
          </ul>
        </section>
      </div>

      <div className="results-footer">
        <p className="report-path">Report saved to: {path}</p>
      </div>
    </div>
  );
}
