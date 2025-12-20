import React, { useRef, useEffect } from 'react';

interface ProgressViewProps {
  progress: string[];
  onCancel: () => void;
}

export function ProgressView({ progress, onCancel }: ProgressViewProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [progress]);

  return (
    <div className="progress-view">
      <div className="progress-header">
        <div className="spinner" />
        <h2>Validating Your Idea...</h2>
        <p>Running JTBD analysis with AI-powered insights</p>
      </div>

      <div className="progress-log" ref={logRef}>
        {progress.map((line, index) => (
          <div key={index} className="log-line">
            {line}
          </div>
        ))}
        {progress.length === 0 && (
          <div className="log-line dim">Initializing analysis pipeline...</div>
        )}
      </div>

      <div className="progress-actions">
        <button onClick={onCancel} className="btn-secondary">
          Cancel
        </button>
      </div>
    </div>
  );
}
