import { create } from 'zustand';

type TaskAxis = 'command' | 'query' | 'navigate' | 'tool';
type ScopeAxis = 'local' | 'project' | 'system' | 'global';
type TimeAxis = 'immediate' | 'short' | 'long' | 'recurring';
type ModeAxis = 'interactive' | 'batch' | 'analytical' | 'learning';

interface NavigationState {
  task: TaskAxis;
  scope: ScopeAxis;
  time: TimeAxis;
  mode: ModeAxis;
  history: Array<{
    task: TaskAxis;
    scope: ScopeAxis;
    time: TimeAxis;
    mode: ModeAxis;
    timestamp: number;
  }>;
  navigate: (axis: 'task' | 'scope' | 'time' | 'mode', value: string) => void;
  reset: () => void;
  getContext: () => string;
}

export const use4DNavigation = create<NavigationState>((set, get) => ({
  task: 'command',
  scope: 'local',
  time: 'immediate',
  mode: 'interactive',
  history: [],

  navigate: (axis, value) => {
    const currentState = get();
    const newState = {
      task: currentState.task,
      scope: currentState.scope,
      time: currentState.time,
      mode: currentState.mode,
      timestamp: Date.now(),
    };

    // Update the specific axis
    switch (axis) {
      case 'task':
        newState.task = value as TaskAxis;
        break;
      case 'scope':
        newState.scope = value as ScopeAxis;
        break;
      case 'time':
        newState.time = value as TimeAxis;
        break;
      case 'mode':
        newState.mode = value as ModeAxis;
        break;
    }

    set({
      ...newState,
      history: [...currentState.history, currentState].slice(-10), // Keep last 10 states
    });
  },

  reset: () => {
    set({
      task: 'command',
      scope: 'local',
      time: 'immediate',
      mode: 'interactive',
      history: [],
    });
  },

  getContext: () => {
    const state = get();
    return `Task:${state.task}|Scope:${state.scope}|Time:${state.time}|Mode:${state.mode}`;
  },
}));