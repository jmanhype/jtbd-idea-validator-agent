import { gql } from '@apollo/client';

// Mutations
export const RUN_TESTS = gql`
  mutation RunTests($suite: String!, $dryRun: Boolean!, $options: TestOptions) {
    runTests(suite: $suite, dryRun: $dryRun, options: $options) {
      id
      suite
      status
      startTime
      dryRun
      results {
        name
        status
        duration
        error
        assertions
      }
    }
  }
`;

export const SCAFFOLD_PAGE = gql`
  mutation ScaffoldPage(
    $template: String!
    $name: String!
    $path: String!
    $dryRun: Boolean!
    $options: ScaffoldOptions
  ) {
    scaffoldPage(
      template: $template
      name: $name
      path: $path
      dryRun: $dryRun
      options: $options
    ) {
      id
      template
      dryRun
      preview
      files {
        path
        action
        content
        diff
      }
    }
  }
`;

export const APPLY_CHANGES = gql`
  mutation ApplyChanges($runId: String!, $confirmation: ConfirmationInput!) {
    applyChanges(runId: $runId, confirmation: $confirmation) {
      success
      message
      appliedChanges {
        path
        action
        content
      }
      errors
    }
  }
`;

// Queries
export const GET_TEST_STATUS = gql`
  query GetTestStatus($runId: String!) {
    testStatus(runId: $runId) {
      id
      suite
      status
      startTime
      endTime
      dryRun
      results {
        name
        status
        duration
        error
        assertions
      }
    }
  }
`;

export const GET_AVAILABLE_TESTS = gql`
  query GetAvailableTests {
    availableTests {
      name
      description
      testCount
      estimatedDuration
    }
  }
`;

export const GET_SCAFFOLD_TEMPLATES = gql`
  query GetScaffoldTemplates {
    scaffoldTemplates {
      name
      description
      category
      variables {
        name
        type
        required
        default
      }
    }
  }
`;

// Subscriptions
export const RUN_PROGRESS_SUBSCRIPTION = gql`
  subscription RunProgress($runId: String!) {
    runProgress(runId: $runId) {
      runId
      type
      progress
      message
      details
      timestamp
    }
  }
`;

export const SYSTEM_EVENTS_SUBSCRIPTION = gql`
  subscription SystemEvents {
    systemEvents {
      id
      type
      payload
      timestamp
    }
  }
`;

// Type definitions for TypeScript
export interface TestOptions {
  parallel?: boolean;
  timeout?: number;
  verbose?: boolean;
  coverage?: boolean;
}

export interface ScaffoldOptions {
  typescript?: boolean;
  cssModule?: boolean;
  tests?: boolean;
  storybook?: boolean;
}

export interface ConfirmationInput {
  confirmed: boolean;
  reason?: string;
  modifications?: Array<{
    fileIndex: number;
    newContent: string;
  }>;
}

export enum TestStatus {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  PASSED = 'PASSED',
  FAILED = 'FAILED',
  SKIPPED = 'SKIPPED',
  CANCELLED = 'CANCELLED',
}

export enum FileAction {
  CREATE = 'CREATE',
  UPDATE = 'UPDATE',
  DELETE = 'DELETE',
}

export enum ProgressType {
  TEST_START = 'TEST_START',
  TEST_COMPLETE = 'TEST_COMPLETE',
  TEST_PROGRESS = 'TEST_PROGRESS',
  SCAFFOLD_START = 'SCAFFOLD_START',
  SCAFFOLD_COMPLETE = 'SCAFFOLD_COMPLETE',
  ERROR = 'ERROR',
  WARNING = 'WARNING',
  INFO = 'INFO',
}