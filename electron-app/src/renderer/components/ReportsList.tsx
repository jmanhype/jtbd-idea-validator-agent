import React, { useState, useEffect } from 'react';

interface Report {
  path: string;
  name: string;
  date: string;
}

interface ReportsListProps {
  onSelect: (path: string) => void;
}

export function ReportsList({ onSelect }: ReportsListProps) {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    setLoading(true);
    try {
      const reportList = await window.jtbdAPI.listReports();
      setReports(reportList);
    } catch (error) {
      console.error('Failed to load reports:', error);
    }
    setLoading(false);
  };

  const formatName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/^mcp /, '')
      .trim();
  };

  if (loading) {
    return (
      <div className="reports-list loading">
        <div className="spinner" />
        <p>Loading reports...</p>
      </div>
    );
  }

  if (reports.length === 0) {
    return (
      <div className="reports-list empty">
        <div className="empty-icon">📊</div>
        <h3>No Reports Yet</h3>
        <p>Run your first idea validation to see results here.</p>
      </div>
    );
  }

  // Group by date
  const grouped = reports.reduce((acc, report) => {
    if (!acc[report.date]) {
      acc[report.date] = [];
    }
    acc[report.date].push(report);
    return acc;
  }, {} as Record<string, Report[]>);

  return (
    <div className="reports-list">
      <div className="reports-header">
        <h2>Validation History</h2>
        <button onClick={loadReports} className="btn-refresh">
          ↻ Refresh
        </button>
      </div>

      {Object.entries(grouped).map(([date, dateReports]) => (
        <div key={date} className="reports-group">
          <h3 className="date-header">{date}</h3>
          <div className="reports-grid">
            {dateReports.map((report) => (
              <button
                key={report.path}
                className="report-card"
                onClick={() => onSelect(report.path)}
              >
                <div className="report-icon">📋</div>
                <div className="report-info">
                  <span className="report-name">{formatName(report.name)}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
