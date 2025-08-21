"""
GraphQL Control Hook for Frontend Integration
Replaces JSON-RPC with Apollo Client GraphQL
"""

import { useCallback, useState } from 'react';
import { ApolloClient, InMemoryCache, gql, useSubscription, useMutation, useQuery } from '@apollo/client';
import { WebSocketLink } from '@apollo/client/link/ws';
import { split, HttpLink } from '@apollo/client';
import { getMainDefinition } from '@apollo/client/utilities';

// GraphQL Queries
const HEALTH_QUERY = gql`
  query GetHealth {
    health {
      overallStatus
      modules {
        moduleName
        status
        latencyMs
        errorCount
        successRate
      }
      activeSessions
      uptimeSeconds
      memoryUsageMb
      cpuPercentage
    }
  }
`;

const LIST_TOOLS_QUERY = gql`
  query ListTools {
    listTools
  }
`;

// GraphQL Mutations
const RUN_TESTS_MUTATION = gql`
  mutation RunTests($input: RunTestsInput!) {
    runTests(input: $input) {
      suiteName
      totalTests
      passed
      failed
      skipped
      durationMs
      executionMode
      dryRunPreview
      testResults {
        testName
        status
        durationMs
        errorMessage
      }
    }
  }
`;

const SCAFFOLD_PAGE_MUTATION = gql`
  mutation ScaffoldPage($input: ScaffoldPageInput!) {
    scaffoldPage(input: $input) {
      pageName
      totalFiles
      totalSizeBytes
      executionMode
      dryRunPreview
      filesCreated {
        filePath
        content
        fileType
        sizeBytes
        wouldOverwrite
      }
      routesModified
    }
  }
`;

const EXECUTE_TOOL_MUTATION = gql`
  mutation ExecuteTool($input: ToolExecutionInput!) {
    executeTool(input: $input) {
      toolName
      category
      executionTimeMs
      result
      executionMode
      dryRunPreview
      cached
    }
  }
`;

const CONFIRM_ACTION_MUTATION = gql`
  mutation ConfirmAction($requestId: String!, $confirmed: Boolean!, $reason: String) {
    confirmAction(requestId: $requestId, confirmed: $confirmed, reason: $reason) {
      requestId
      confirmed
      confirmedBy
      confirmedAt
      rejectionReason
    }
  }
`;

// GraphQL Subscriptions
const PROGRESS_SUBSCRIPTION = gql`
  subscription OnProgressUpdate($taskId: String!) {
    progressUpdates(taskId: $taskId) {
      taskId
      progress {
        status
        progressPercentage
        currentStep
        totalSteps
        completedSteps
        startTime
        estimatedCompletion
        logs
        metrics
      }
      timestamp
    }
  }
`;

const LOGS_SUBSCRIPTION = gql`
  subscription OnSystemLogs($level: String) {
    systemLogs(level: $level) {
      level
      message
      module
      timestamp
      metadata
    }
  }
`;

// Apollo Client setup
const httpLink = new HttpLink({
  uri: 'http://localhost:8001/graphql',
});

const wsLink = new WebSocketLink({
  uri: 'ws://localhost:8001/graphql',
  options: {
    reconnect: true,
  },
});

const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,
  httpLink
);

const apolloClient = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'network-only',
    },
  },
});

interface UseGraphQLControlProps {
  defaultDryRun?: boolean;
  requireConfirmation?: boolean;
  autoRetry?: boolean;
}

export function useGraphQLControl({
  defaultDryRun = true,
  requireConfirmation = true,
  autoRetry = false,
}: UseGraphQLControlProps = {}) {
  const [pendingConfirmations, setPendingConfirmations] = useState<Map<string, any>>(new Map());
  
  // Queries
  const { data: healthData, loading: healthLoading, refetch: refetchHealth } = useQuery(HEALTH_QUERY, {
    client: apolloClient,
    pollInterval: 5000, // Poll every 5 seconds
  });
  
  const { data: toolsData, loading: toolsLoading, refetch: refetchTools } = useQuery(LIST_TOOLS_QUERY, {
    client: apolloClient,
  });
  
  // Mutations
  const [runTestsMutation] = useMutation(RUN_TESTS_MUTATION, { client: apolloClient });
  const [scaffoldPageMutation] = useMutation(SCAFFOLD_PAGE_MUTATION, { client: apolloClient });
  const [executeToolMutation] = useMutation(EXECUTE_TOOL_MUTATION, { client: apolloClient });
  const [confirmActionMutation] = useMutation(CONFIRM_ACTION_MUTATION, { client: apolloClient });
  
  // Run tests with safety controls
  const runTests = useCallback(async (
    testSuite: string,
    options: {
      pattern?: string;
      mode?: 'DRY_RUN' | 'APPLY';
      parallel?: boolean;
      verbose?: boolean;
      timeout?: number;
    } = {}
  ) => {
    const mode = options.mode || (defaultDryRun ? 'DRY_RUN' : 'APPLY');
    
    // First, always do a dry run if not explicitly set
    if (mode === 'APPLY' && requireConfirmation) {
      const dryRunResult = await runTestsMutation({
        variables: {
          input: {
            testSuite,
            testPattern: options.pattern || '*',
            mode: 'DRY_RUN',
            parallel: options.parallel ?? true,
            verbose: options.verbose ?? false,
            timeoutSeconds: options.timeout || 300,
          },
        },
      });
      
      // Show dry run preview and request confirmation
      const confirmationId = `test_${Date.now()}`;
      setPendingConfirmations(prev => new Map(prev).set(confirmationId, {
        type: 'runTests',
        preview: dryRunResult.data.runTests.dryRunPreview,
        action: async () => {
          // Execute actual tests
          return runTestsMutation({
            variables: {
              input: {
                testSuite,
                testPattern: options.pattern || '*',
                mode: 'APPLY',
                parallel: options.parallel ?? true,
                verbose: options.verbose ?? false,
                timeoutSeconds: options.timeout || 300,
              },
            },
          });
        },
      }));
      
      return { confirmationRequired: true, confirmationId, preview: dryRunResult.data.runTests };
    }
    
    // Direct execution
    const result = await runTestsMutation({
      variables: {
        input: {
          testSuite,
          testPattern: options.pattern || '*',
          mode,
          parallel: options.parallel ?? true,
          verbose: options.verbose ?? false,
          timeoutSeconds: options.timeout || 300,
        },
      },
    });
    
    return result.data.runTests;
  }, [runTestsMutation, defaultDryRun, requireConfirmation]);
  
  // Scaffold page with safety controls
  const scaffoldPage = useCallback(async (
    pageName: string,
    pageType: string,
    components: string[],
    options: {
      route?: string;
      mode?: 'DRY_RUN' | 'APPLY';
      useTypescript?: boolean;
      includeTests?: boolean;
      stylingFramework?: string;
    } = {}
  ) => {
    const mode = options.mode || (defaultDryRun ? 'DRY_RUN' : 'APPLY');
    
    if (mode === 'APPLY' && requireConfirmation) {
      const dryRunResult = await scaffoldPageMutation({
        variables: {
          input: {
            pageName,
            pageType,
            components,
            route: options.route || `/${pageName.toLowerCase()}`,
            mode: 'DRY_RUN',
            useTypescript: options.useTypescript ?? true,
            includeTests: options.includeTests ?? true,
            stylingFramework: options.stylingFramework || 'tailwind',
          },
        },
      });
      
      const confirmationId = `scaffold_${Date.now()}`;
      setPendingConfirmations(prev => new Map(prev).set(confirmationId, {
        type: 'scaffoldPage',
        preview: dryRunResult.data.scaffoldPage.dryRunPreview,
        action: async () => {
          return scaffoldPageMutation({
            variables: {
              input: {
                pageName,
                pageType,
                components,
                route: options.route || `/${pageName.toLowerCase()}`,
                mode: 'APPLY',
                useTypescript: options.useTypescript ?? true,
                includeTests: options.includeTests ?? true,
                stylingFramework: options.stylingFramework || 'tailwind',
              },
            },
          });
        },
      }));
      
      return { confirmationRequired: true, confirmationId, preview: dryRunResult.data.scaffoldPage };
    }
    
    const result = await scaffoldPageMutation({
      variables: {
        input: {
          pageName,
          pageType,
          components,
          route: options.route || `/${pageName.toLowerCase()}`,
          mode,
          useTypescript: options.useTypescript ?? true,
          includeTests: options.includeTests ?? true,
          stylingFramework: options.stylingFramework || 'tailwind',
        },
      },
    });
    
    return result.data.scaffoldPage;
  }, [scaffoldPageMutation, defaultDryRun, requireConfirmation]);
  
  // Execute tool (replacement for JSON-RPC)
  const executeTool = useCallback(async (
    toolName: string,
    parameters: any,
    options: {
      mode?: 'DRY_RUN' | 'APPLY';
      timeout?: number;
    } = {}
  ) => {
    const mode = options.mode || (defaultDryRun ? 'DRY_RUN' : 'APPLY');
    
    const result = await executeToolMutation({
      variables: {
        input: {
          toolName,
          parameters,
          mode,
          timeoutMs: options.timeout || 5000,
        },
      },
    });
    
    return result.data.executeTool;
  }, [executeToolMutation, defaultDryRun]);
  
  // Confirm pending action
  const confirmAction = useCallback(async (confirmationId: string, confirmed: boolean, reason?: string) => {
    const pending = pendingConfirmations.get(confirmationId);
    if (!pending) {
      throw new Error(`No pending confirmation with ID: ${confirmationId}`);
    }
    
    if (confirmed && pending.action) {
      // Execute the pending action
      const result = await pending.action();
      setPendingConfirmations(prev => {
        const next = new Map(prev);
        next.delete(confirmationId);
        return next;
      });
      return result;
    } else {
      // Rejection
      setPendingConfirmations(prev => {
        const next = new Map(prev);
        next.delete(confirmationId);
        return next;
      });
      return { confirmed: false, reason };
    }
  }, [pendingConfirmations]);
  
  // Subscribe to progress updates
  const useProgressUpdates = (taskId: string) => {
    return useSubscription(PROGRESS_SUBSCRIPTION, {
      variables: { taskId },
      client: apolloClient,
    });
  };
  
  // Subscribe to system logs
  const useSystemLogs = (level?: string) => {
    return useSubscription(LOGS_SUBSCRIPTION, {
      variables: { level },
      client: apolloClient,
    });
  };
  
  return {
    // Data
    health: healthData?.health,
    tools: toolsData?.listTools,
    pendingConfirmations: Array.from(pendingConfirmations.entries()).map(([id, data]) => ({
      id,
      ...data,
    })),
    
    // Loading states
    healthLoading,
    toolsLoading,
    
    // Actions
    runTests,
    scaffoldPage,
    executeTool,
    confirmAction,
    refetchHealth,
    refetchTools,
    
    // Subscriptions
    useProgressUpdates,
    useSystemLogs,
    
    // Apollo Client (for advanced usage)
    apolloClient,
  };
}