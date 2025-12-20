interface IdeaInput {
  idea_id: string;
  title: string;
  problem_statement: string;
  solution_overview: string;
  hunches: string[];
  target_customer?: string;
  competitive_landscape?: string[];
  revenue_streams?: string[];
  value_propositions?: string[];
  constraints?: string[];
  risks_and_challenges?: string[];
  context?: Record<string, string>;
}

interface ValidationResult {
  success: boolean;
  output: string;
  reportPath?: string;
}

interface Report {
  path: string;
  name: string;
  date: string;
}

interface JtbdAPI {
  validate: (ideaData: IdeaInput) => Promise<ValidationResult>;
  cancel: () => Promise<{ cancelled: boolean }>;
  loadReport: (reportPath: string) => Promise<{ success: boolean; content?: string; error?: string }>;
  listReports: () => Promise<Report[]>;
  onProgress: (callback: (progress: string) => void) => () => void;
  saveDialog: (options: Electron.SaveDialogOptions) => Promise<Electron.SaveDialogReturnValue>;
}

declare global {
  interface Window {
    jtbdAPI: JtbdAPI;
  }
}

export {};
