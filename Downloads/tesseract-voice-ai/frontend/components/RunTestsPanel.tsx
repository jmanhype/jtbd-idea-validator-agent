'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useMutation, useSubscription, useQuery } from '@apollo/client';
import {
  RUN_TESTS,
  APPLY_CHANGES,
  RUN_PROGRESS_SUBSCRIPTION,
  GET_AVAILABLE_TESTS,
  TestStatus,
  TestOptions,
  ProgressType,
} from '../lib/graphql-operations';

interface RunTestsPanelProps {
  className?: string;
}

export const RunTestsPanel: React.FC<RunTestsPanelProps> = ({ className }) => {
  const [selectedSuite, setSelectedSuite] = useState<string>('unit');
  const [isDryRun, setIsDryRun] = useState<boolean>(true);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [testResults, setTestResults] = useState<any[]>([]);
  const [showConfirmDialog, setShowConfirmDialog] = useState<boolean>(false);
  
  // GraphQL hooks
  const { data: availableTests } = useQuery(GET_AVAILABLE_TESTS);
  const [runTests, { loading: runningTests }] = useMutation(RUN_TESTS);
  const [applyChanges, { loading: applyingChanges }] = useMutation(APPLY_CHANGES);
  
  // Subscribe to progress updates when we have a run ID
  const { data: progressData } = useSubscription(RUN_PROGRESS_SUBSCRIPTION, {
    variables: { runId: currentRunId! },
    skip: !currentRunId,
  });
  
  // Update progress when subscription data arrives
  useEffect(() => {
    if (progressData?.runProgress) {
      const update = progressData.runProgress;
      setProgress(update.progress);
      setStatusMessage(update.message);
      
      if (update.type === ProgressType.TEST_COMPLETE) {
        // Test run completed
        setCurrentRunId(null);
      }
    }
  }, [progressData]);
  
  // Handle test execution
  const handleRunTests = useCallback(async () => {
    try {
      const options: TestOptions = {
        parallel: true,
        timeout: 30000,
        verbose: true,
        coverage: false,
      };
      
      const { data } = await runTests({
        variables: {
          suite: selectedSuite,
          dryRun: isDryRun,
          options,
        },
      });
      
      if (data?.runTests) {
        const run = data.runTests;
        setCurrentRunId(run.id);
        setTestResults(run.results);
        
        if (isDryRun) {
          // Show confirmation dialog for dry run
          setShowConfirmDialog(true);
          setStatusMessage(`Dry run complete. ${run.results.length} tests would be executed.`);
        } else {
          setStatusMessage('Tests running...');
        }
      }
    } catch (error) {
      console.error('Error running tests:', error);
      setStatusMessage('Error: Failed to start test run');
    }
  }, [selectedSuite, isDryRun, runTests]);
  
  // Handle applying changes after dry run
  const handleApplyChanges = useCallback(async (confirmed: boolean) => {
    if (!currentRunId) return;
    
    try {
      const { data } = await applyChanges({
        variables: {
          runId: currentRunId,
          confirmation: {
            confirmed,
            reason: confirmed ? null : 'User cancelled',
          },
        },
      });
      
      if (data?.applyChanges) {
        const result = data.applyChanges;
        if (result.success) {
          setStatusMessage(result.message);
          if (confirmed) {
            // Tests will now actually run
            setProgress(0);
          } else {
            // Reset state
            setCurrentRunId(null);
            setTestResults([]);
          }
        } else {
          setStatusMessage(`Error: ${result.message}`);
        }
      }
    } catch (error) {
      console.error('Error applying changes:', error);
      setStatusMessage('Error: Failed to apply changes');
    }
    
    setShowConfirmDialog(false);
  }, [currentRunId, applyChanges]);
  
  // Get status color
  const getStatusColor = (status: TestStatus) => {
    switch (status) {
      case TestStatus.PASSED:
        return 'text-green-600';
      case TestStatus.FAILED:
        return 'text-red-600';
      case TestStatus.RUNNING:
        return 'text-blue-600';
      case TestStatus.PENDING:
        return 'text-gray-500';
      case TestStatus.SKIPPED:
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };
  
  return (
    <div className={`p-6 bg-white rounded-lg shadow-lg ${className}`}>
      <h2 className="text-2xl font-bold mb-4">Test Control Panel</h2>
      
      {/* Test Suite Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Select Test Suite</label>
        <select
          value={selectedSuite}
          onChange={(e) => setSelectedSuite(e.target.value)}
          className="w-full p-2 border rounded-md"
          disabled={runningTests || applyingChanges}
        >
          {availableTests?.availableTests.map((suite: any) => (
            <option key={suite.name} value={suite.name}>
              {suite.name} - {suite.description} ({suite.testCount} tests)
            </option>
          ))}
        </select>
      </div>
      
      {/* Dry Run Toggle */}
      <div className="mb-4 flex items-center">
        <input
          type="checkbox"
          id="dryRun"
          checked={isDryRun}
          onChange={(e) => setIsDryRun(e.target.checked)}
          className="mr-2"
          disabled={runningTests || applyingChanges}
        />
        <label htmlFor="dryRun" className="text-sm font-medium">
          Dry Run Mode (preview without executing)
        </label>
      </div>
      
      {/* Run Button */}
      <button
        onClick={handleRunTests}
        disabled={runningTests || applyingChanges || !!currentRunId}
        className="mb-4 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
      >
        {runningTests ? 'Starting...' : isDryRun ? 'Preview Tests' : 'Run Tests'}
      </button>
      
      {/* Progress Bar */}
      {currentRunId && !isDryRun && (
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
      
      {/* Status Message */}
      {statusMessage && (
        <div className="mb-4 p-3 bg-gray-100 rounded-md">
          <p className="text-sm">{statusMessage}</p>
        </div>
      )}
      
      {/* Test Results */}
      {testResults.length > 0 && (
        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2">Test Results</h3>
          <div className="space-y-2">
            {testResults.map((result, index) => (
              <div key={index} className="flex justify-between p-2 bg-gray-50 rounded">
                <span className="font-mono text-sm">{result.name}</span>
                <span className={`text-sm font-medium ${getStatusColor(result.status)}`}>
                  {result.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Confirmation Dialog */}
      {showConfirmDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg max-w-md">
            <h3 className="text-lg font-bold mb-4">Confirm Test Execution</h3>
            <p className="mb-4">
              Dry run complete. {testResults.length} tests are ready to execute.
              Do you want to proceed with the actual test run?
            </p>
            <div className="flex space-x-4">
              <button
                onClick={() => handleApplyChanges(true)}
                disabled={applyingChanges}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                {applyingChanges ? 'Applying...' : 'Execute Tests'}
              </button>
              <button
                onClick={() => handleApplyChanges(false)}
                disabled={applyingChanges}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RunTestsPanel;