import React, { useState, useEffect } from 'react';
import { useGraphQLControl } from '../hooks/useGraphQLControl';
import { AlertCircle, CheckCircle, XCircle, Loader2, Play, Eye } from 'lucide-react';

interface ConfirmationDialogProps {
  confirmation: {
    id: string;
    type: string;
    preview: string;
  };
  onConfirm: (id: string, confirmed: boolean, reason?: string) => void;
}

const ConfirmationDialog: React.FC<ConfirmationDialogProps> = ({ confirmation, onConfirm }) => {
  const [reason, setReason] = useState('');
  
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-yellow-500" />
          Confirmation Required
        </h3>
        
        <div className="mb-4">
          <p className="text-sm text-gray-600 mb-2">Operation: {confirmation.type}</p>
          <div className="bg-gray-50 rounded p-4 font-mono text-sm whitespace-pre-wrap">
            {confirmation.preview}
          </div>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Rejection Reason (optional)
          </label>
          <input
            type="text"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
            placeholder="Why are you rejecting this operation?"
          />
        </div>
        
        <div className="flex gap-3 justify-end">
          <button
            onClick={() => onConfirm(confirmation.id, false, reason)}
            className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 flex items-center gap-2"
          >
            <XCircle className="w-4 h-4" />
            Reject
          </button>
          <button
            onClick={() => onConfirm(confirmation.id, true)}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 flex items-center gap-2"
          >
            <CheckCircle className="w-4 h-4" />
            Apply Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export function GraphQLControlPanel() {
  const {
    health,
    tools,
    pendingConfirmations,
    healthLoading,
    toolsLoading,
    runTests,
    scaffoldPage,
    executeTool,
    confirmAction,
    useProgressUpdates,
    useSystemLogs,
  } = useGraphQLControl({
    defaultDryRun: true,
    requireConfirmation: true,
  });
  
  const [activeTab, setActiveTab] = useState<'tests' | 'scaffold' | 'tools' | 'health'>('health');
  const [testSuite, setTestSuite] = useState('all');
  const [pageName, setPageName] = useState('');
  const [selectedTool, setSelectedTool] = useState('');
  const [toolParams, setToolParams] = useState('{}');
  const [executionMode, setExecutionMode] = useState<'DRY_RUN' | 'APPLY'>('DRY_RUN');
  const [results, setResults] = useState<any[]>([]);
  
  // Subscribe to system logs
  const { data: logsData } = useSystemLogs();
  
  useEffect(() => {
    if (logsData) {
      console.log('System log:', logsData.systemLogs);
    }
  }, [logsData]);
  
  const handleRunTests = async () => {
    try {
      const result = await runTests(testSuite, {
        mode: executionMode,
        parallel: true,
        verbose: true,
      });
      
      if (result.confirmationRequired) {
        // Confirmation dialog will appear
        console.log('Confirmation required for test execution');
      } else {
        setResults(prev => [...prev, { type: 'test', data: result, timestamp: new Date() }]);
      }
    } catch (error) {
      console.error('Test execution failed:', error);
    }
  };
  
  const handleScaffoldPage = async () => {
    if (!pageName) return;
    
    try {
      const result = await scaffoldPage(
        pageName,
        'dashboard',
        ['Header', 'Content', 'Footer'],
        {
          mode: executionMode,
          useTypescript: true,
          includeTests: true,
        }
      );
      
      if (result.confirmationRequired) {
        console.log('Confirmation required for scaffolding');
      } else {
        setResults(prev => [...prev, { type: 'scaffold', data: result, timestamp: new Date() }]);
      }
    } catch (error) {
      console.error('Scaffolding failed:', error);
    }
  };
  
  const handleExecuteTool = async () => {
    if (!selectedTool) return;
    
    try {
      const params = JSON.parse(toolParams);
      const result = await executeTool(selectedTool, params, {
        mode: executionMode,
        timeout: 5000,
      });
      
      setResults(prev => [...prev, { type: 'tool', data: result, timestamp: new Date() }]);
    } catch (error) {
      console.error('Tool execution failed:', error);
    }
  };
  
  const handleConfirmation = async (id: string, confirmed: boolean, reason?: string) => {
    try {
      const result = await confirmAction(id, confirmed, reason);
      if (result && confirmed) {
        setResults(prev => [...prev, { type: 'confirmed', data: result, timestamp: new Date() }]);
      }
    } catch (error) {
      console.error('Confirmation failed:', error);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">GraphQL Control Plane</h1>
        
        {/* Safety Mode Indicator */}
        <div className="mb-6 p-4 bg-white rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium">Execution Mode:</span>
              <div className="flex gap-2">
                <button
                  onClick={() => setExecutionMode('DRY_RUN')}
                  className={`px-3 py-1 rounded-md text-sm ${
                    executionMode === 'DRY_RUN'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  <Eye className="w-4 h-4 inline mr-1" />
                  DRY RUN (Safe)
                </button>
                <button
                  onClick={() => setExecutionMode('APPLY')}
                  className={`px-3 py-1 rounded-md text-sm ${
                    executionMode === 'APPLY'
                      ? 'bg-green-500 text-white'
                      : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  <Play className="w-4 h-4 inline mr-1" />
                  APPLY (Requires Confirmation)
                </button>
              </div>
            </div>
            
            {pendingConfirmations.length > 0 && (
              <div className="flex items-center gap-2 text-yellow-600">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm font-medium">
                  {pendingConfirmations.length} pending confirmation(s)
                </span>
              </div>
            )}
          </div>
        </div>
        
        {/* Tabs */}
        <div className="mb-6">
          <div className="flex gap-2">
            {(['health', 'tests', 'scaffold', 'tools'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-md capitalize ${
                  activeTab === tab
                    ? 'bg-white shadow-sm font-medium'
                    : 'text-gray-600 hover:bg-white/50'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
        
        {/* Tab Content */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          {activeTab === 'health' && (
            <div>
              <h2 className="text-xl font-semibold mb-4">System Health</h2>
              {healthLoading ? (
                <Loader2 className="w-6 h-6 animate-spin" />
              ) : health ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-600">Status</p>
                      <p className="text-lg font-semibold">{health.overallStatus}</p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-600">Active Sessions</p>
                      <p className="text-lg font-semibold">{health.activeSessions}</p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-600">Memory</p>
                      <p className="text-lg font-semibold">{health.memoryUsageMb.toFixed(1)} MB</p>
                    </div>
                    <div className="p-3 bg-gray-50 rounded">
                      <p className="text-sm text-gray-600">CPU</p>
                      <p className="text-lg font-semibold">{health.cpuPercentage.toFixed(1)}%</p>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="font-medium mb-2">Modules</h3>
                    <div className="space-y-2">
                      {health.modules.map((module) => (
                        <div key={module.moduleName} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                          <span className="font-mono text-sm">{module.moduleName}</span>
                          <div className="flex items-center gap-4">
                            <span className="text-sm text-gray-600">{module.latencyMs.toFixed(1)}ms</span>
                            <span className={`px-2 py-1 rounded text-xs ${
                              module.status === 'ready' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                            }`}>
                              {module.status}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <p>No health data available</p>
              )}
            </div>
          )}
          
          {activeTab === 'tests' && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Run Tests</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Test Suite</label>
                  <input
                    type="text"
                    value={testSuite}
                    onChange={(e) => setTestSuite(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., all, unit, integration"
                  />
                </div>
                <button
                  onClick={handleRunTests}
                  className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                >
                  Run Tests ({executionMode})
                </button>
              </div>
            </div>
          )}
          
          {activeTab === 'scaffold' && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Scaffold Page</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Page Name</label>
                  <input
                    type="text"
                    value={pageName}
                    onChange={(e) => setPageName(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., Dashboard, Settings, Profile"
                  />
                </div>
                <button
                  onClick={handleScaffoldPage}
                  className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                  disabled={!pageName}
                >
                  Scaffold Page ({executionMode})
                </button>
              </div>
            </div>
          )}
          
          {activeTab === 'tools' && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Execute Tools</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Tool</label>
                  <select
                    value={selectedTool}
                    onChange={(e) => setSelectedTool(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="">Select a tool...</option>
                    {tools?.map((tool) => (
                      <option key={tool} value={tool}>{tool}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Parameters (JSON)</label>
                  <textarea
                    value={toolParams}
                    onChange={(e) => setToolParams(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md font-mono text-sm"
                    rows={4}
                    placeholder='{"param1": "value1"}'
                  />
                </div>
                <button
                  onClick={handleExecuteTool}
                  className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                  disabled={!selectedTool}
                >
                  Execute Tool ({executionMode})
                </button>
              </div>
            </div>
          )}
        </div>
        
        {/* Results */}
        {results.length > 0 && (
          <div className="mt-6 bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">Execution Results</h2>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {results.map((result, index) => (
                <div key={index} className="p-4 bg-gray-50 rounded">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium capitalize">{result.type}</span>
                    <span className="text-xs text-gray-500">
                      {result.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <pre className="text-xs overflow-x-auto">
                    {JSON.stringify(result.data, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Confirmation Dialogs */}
        {pendingConfirmations.map((confirmation) => (
          <ConfirmationDialog
            key={confirmation.id}
            confirmation={confirmation}
            onConfirm={handleConfirmation}
          />
        ))}
      </div>
    </div>
  );
}