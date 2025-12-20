import { contextBridge, ipcRenderer } from 'electron';

export interface IdeaInput {
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

export interface ValidationResult {
  success: boolean;
  output: string;
  reportPath?: string;
}

export interface Report {
  path: string;
  name: string;
  date: string;
}

contextBridge.exposeInMainWorld('jtbdAPI', {
  validate: (ideaData: IdeaInput): Promise<ValidationResult> =>
    ipcRenderer.invoke('jtbd:validate', ideaData),

  cancel: (): Promise<{ cancelled: boolean }> =>
    ipcRenderer.invoke('jtbd:cancel'),

  loadReport: (reportPath: string): Promise<{ success: boolean; content?: string; error?: string }> =>
    ipcRenderer.invoke('jtbd:load-report', reportPath),

  listReports: (): Promise<Report[]> =>
    ipcRenderer.invoke('jtbd:list-reports'),

  onProgress: (callback: (progress: string) => void): (() => void) => {
    const handler = (_event: Electron.IpcRendererEvent, data: string) => callback(data);
    ipcRenderer.on('jtbd:progress', handler);
    return () => ipcRenderer.removeListener('jtbd:progress', handler);
  },

  saveDialog: (options: Electron.SaveDialogOptions): Promise<Electron.SaveDialogReturnValue> =>
    ipcRenderer.invoke('dialog:save', options),
});
