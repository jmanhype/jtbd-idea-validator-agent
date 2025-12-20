import React, { useState } from 'react';

interface IdeaFormProps {
  onSubmit: (data: IdeaData) => void;
}

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

export function IdeaForm({ onSubmit }: IdeaFormProps) {
  const [title, setTitle] = useState('');
  const [problem, setProblem] = useState('');
  const [solution, setSolution] = useState('');
  const [hunches, setHunches] = useState(['', '', '', '']);
  const [targetCustomer, setTargetCustomer] = useState('');
  const [competitors, setCompetitors] = useState('');
  const [revenue, setRevenue] = useState('');
  const [valueProps, setValueProps] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const ideaId = `urn:idea:${Date.now().toString(36)}`;
    const filteredHunches = hunches.filter((h) => h.trim());

    const data: IdeaData = {
      idea_id: ideaId,
      title: title.trim(),
      problem_statement: problem.trim(),
      solution_overview: solution.trim(),
      hunches: filteredHunches.length > 0 ? filteredHunches : ['Market need exists'],
    };

    if (targetCustomer.trim()) {
      data.target_customer = targetCustomer.trim();
    }
    if (competitors.trim()) {
      data.competitive_landscape = competitors.split(',').map((c) => c.trim()).filter(Boolean);
    }
    if (revenue.trim()) {
      data.revenue_streams = revenue.split(',').map((r) => r.trim()).filter(Boolean);
    }
    if (valueProps.trim()) {
      data.value_propositions = valueProps.split(',').map((v) => v.trim()).filter(Boolean);
    }

    onSubmit(data);
  };

  const updateHunch = (index: number, value: string) => {
    const newHunches = [...hunches];
    newHunches[index] = value;
    setHunches(newHunches);
  };

  return (
    <form className="idea-form" onSubmit={handleSubmit}>
      <div className="form-section">
        <h2>Business Idea</h2>

        <div className="form-group">
          <label htmlFor="title">Idea Title *</label>
          <input
            id="title"
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g., AI-powered fitness coaching app"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="problem">Problem Statement *</label>
          <textarea
            id="problem"
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            placeholder="What problem are you solving? Who has this problem?"
            rows={3}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="solution">Solution Overview *</label>
          <textarea
            id="solution"
            value={solution}
            onChange={(e) => setSolution(e.target.value)}
            placeholder="How does your solution work? What makes it unique?"
            rows={3}
            required
          />
        </div>
      </div>

      <div className="form-section">
        <h2>Hypotheses (Hunches)</h2>
        <p className="form-hint">
          What assumptions are you making about the market, customers, or solution?
        </p>

        {hunches.map((hunch, index) => (
          <div className="form-group" key={index}>
            <label htmlFor={`hunch-${index}`}>Hunch {index + 1}</label>
            <input
              id={`hunch-${index}`}
              type="text"
              value={hunch}
              onChange={(e) => updateHunch(index, e.target.value)}
              placeholder={`e.g., ${
                index === 0
                  ? 'Users will pay for premium features'
                  : index === 1
                  ? 'Word of mouth will drive growth'
                  : index === 2
                  ? 'Mobile-first approach is essential'
                  : 'Existing solutions are too complex'
              }`}
            />
          </div>
        ))}
      </div>

      <div className="form-section">
        <button
          type="button"
          className="toggle-advanced"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? '- Hide Advanced' : '+ Show Advanced Options'}
        </button>

        {showAdvanced && (
          <>
            <div className="form-group">
              <label htmlFor="target">Target Customer</label>
              <input
                id="target"
                type="text"
                value={targetCustomer}
                onChange={(e) => setTargetCustomer(e.target.value)}
                placeholder="e.g., Busy professionals aged 25-45"
              />
            </div>

            <div className="form-group">
              <label htmlFor="competitors">Competitors (comma-separated)</label>
              <input
                id="competitors"
                type="text"
                value={competitors}
                onChange={(e) => setCompetitors(e.target.value)}
                placeholder="e.g., Competitor A, Competitor B, Generic solutions"
              />
            </div>

            <div className="form-group">
              <label htmlFor="revenue">Revenue Streams (comma-separated)</label>
              <input
                id="revenue"
                type="text"
                value={revenue}
                onChange={(e) => setRevenue(e.target.value)}
                placeholder="e.g., Subscription, Enterprise licensing, API access"
              />
            </div>

            <div className="form-group">
              <label htmlFor="valueProps">Value Propositions (comma-separated)</label>
              <input
                id="valueProps"
                type="text"
                value={valueProps}
                onChange={(e) => setValueProps(e.target.value)}
                placeholder="e.g., Saves 10 hours/week, 50% cost reduction"
              />
            </div>
          </>
        )}
      </div>

      <div className="form-actions">
        <button type="submit" className="btn-primary">
          Validate Idea
        </button>
      </div>
    </form>
  );
}
