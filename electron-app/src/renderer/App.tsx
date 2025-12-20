import React, { useState, useEffect } from 'react';
import { IdeaForm } from './components/IdeaForm';
import { ResultsView } from './components/ResultsView';
import { ReportsList } from './components/ReportsList';
import { ProgressView } from './components/ProgressView';
import './styles/app.css';

type View = 'form' | 'validating' | 'results' | 'history';

interface IdeaData {
  idea_id: string;
  title: string;
  problem_statement: string;
  solution_overview: string;
  hunches: string[];
  target_customer?: string;
  competitive_landscape?: string[];
  revenue_streams?: string[];
  value_propositions?: string[];
}

export default function App() {
  const [view, setView] = useState<View>('form');
  const [progress, setProgress] = useState<string[]>([]);
  const [result, setResult] = useState<{ content: string; path: string } | null>(null);

  useEffect(() => {
    const cleanup = window.jtbdAPI.onProgress((data) => {
      setProgress((prev) => [...prev, data]);
    });
    return cleanup;
  }, []);

  const handleValidate = async (ideaData: IdeaData) => {
    setView('validating');
    setProgress([]);

    try {
      const result = await window.jtbdAPI.validate(ideaData);

      if (result.success && result.reportPath) {
        const report = await window.jtbdAPI.loadReport(result.reportPath);
        if (report.success && report.content) {
          setResult({ content: report.content, path: result.reportPath });
          setView('results');
        }
      }
    } catch (error) {
      console.error('Validation failed:', error);
      setProgress((prev) => [...prev, `Error: ${(error as Error).message}`]);
    }
  };

  const handleCancel = async () => {
    await window.jtbdAPI.cancel();
    setView('form');
    setProgress([]);
  };

  const handleLoadReport = async (path: string) => {
    const report = await window.jtbdAPI.loadReport(path);
    if (report.success && report.content) {
      setResult({ content: report.content, path });
      setView('results');
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>JTBD Idea Validator</h1>
        <nav>
          <button
            className={view === 'form' ? 'active' : ''}
            onClick={() => setView('form')}
            disabled={view === 'validating'}
          >
            New Analysis
          </button>
          <button
            className={view === 'history' ? 'active' : ''}
            onClick={() => setView('history')}
            disabled={view === 'validating'}
          >
            History
          </button>
        </nav>
      </header>

      <main className="app-main">
        {view === 'form' && <IdeaForm onSubmit={handleValidate} />}
        {view === 'validating' && (
          <ProgressView progress={progress} onCancel={handleCancel} />
        )}
        {view === 'results' && result && (
          <ResultsView
            content={result.content}
            path={result.path}
            onBack={() => setView('form')}
          />
        )}
        {view === 'history' && (
          <ReportsList onSelect={handleLoadReport} />
        )}
      </main>
    </div>
  );
}
