import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;

const isDev = process.env.NODE_ENV !== 'production';
const PYTHON_PATH = path.join(__dirname, '../../.venv/bin/python');
const SCRIPT_PATH = path.join(__dirname, '../../run_direct.py');

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#1a1a2e',
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers for JTBD validation
ipcMain.handle('jtbd:validate', async (_event, ideaData) => {
  return new Promise((resolve, reject) => {
    // Create temporary JSON file for the idea
    const tempPath = path.join(app.getPath('temp'), `jtbd-idea-${Date.now()}.json`);
    fs.writeFileSync(tempPath, JSON.stringify(ideaData, null, 2));

    const args = [SCRIPT_PATH, tempPath];

    pythonProcess = spawn(PYTHON_PATH, args, {
      cwd: path.join(__dirname, '../..'),
      env: {
        ...process.env,
        GROQ_API_KEY: process.env.GROQ_API_KEY || '',
        JTBD_DSPY_MODEL: 'groq/llama-3.3-70b-versatile',
      },
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout?.on('data', (data) => {
      stdout += data.toString();
      mainWindow?.webContents.send('jtbd:progress', data.toString());
    });

    pythonProcess.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      fs.unlinkSync(tempPath); // Clean up temp file

      if (code === 0) {
        // Parse the output to find the report path
        const reportMatch = stdout.match(/saved to: (.+\.md)/);
        const reportPath = reportMatch ? reportMatch[1] : null;

        resolve({
          success: true,
          output: stdout,
          reportPath,
        });
      } else {
        reject(new Error(stderr || 'Validation failed'));
      }
    });

    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
});

ipcMain.handle('jtbd:cancel', async () => {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
    return { cancelled: true };
  }
  return { cancelled: false };
});

ipcMain.handle('jtbd:load-report', async (_event, reportPath) => {
  try {
    const content = fs.readFileSync(reportPath, 'utf-8');
    return { success: true, content };
  } catch (err) {
    return { success: false, error: (err as Error).message };
  }
});

ipcMain.handle('jtbd:list-reports', async () => {
  const reportsDir = path.join(__dirname, '../../reports');
  const reports: Array<{ path: string; name: string; date: string }> = [];

  if (!fs.existsSync(reportsDir)) {
    return reports;
  }

  const dateDirs = fs.readdirSync(reportsDir).filter((d) =>
    fs.statSync(path.join(reportsDir, d)).isDirectory()
  );

  for (const dateDir of dateDirs.slice(-7).reverse()) {
    const datePath = path.join(reportsDir, dateDir);
    const reportDirs = fs.readdirSync(datePath).filter((d) =>
      fs.statSync(path.join(datePath, d)).isDirectory()
    );

    for (const reportDir of reportDirs.reverse()) {
      const gammaPath = path.join(datePath, reportDir, 'gamma', 'presentation.md');
      if (fs.existsSync(gammaPath)) {
        reports.push({
          path: gammaPath,
          name: reportDir.replace(/_\d{8}_\d{6}$/, ''),
          date: dateDir,
        });
      }
      if (reports.length >= 20) break;
    }
    if (reports.length >= 20) break;
  }

  return reports;
});

ipcMain.handle('dialog:save', async (_event, options) => {
  if (!mainWindow) return { canceled: true };
  return dialog.showSaveDialog(mainWindow, options);
});
