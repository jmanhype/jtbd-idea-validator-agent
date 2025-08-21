'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Settings, Activity, Layers, Clock, Grid3x3 } from 'lucide-react';
import WaveSurfer from 'wavesurfer.js';
import { useVoiceSession } from '@/hooks/useVoiceSession';
import { use4DNavigation } from '@/hooks/use4DNavigation';
import { NavigationAxis } from '@/components/NavigationAxis';
import { ModuleStatus } from '@/components/ModuleStatus';

export default function TesseractVoiceAI() {
  const [isListening, setIsListening] = useState(false);
  const [currentMode, setCurrentMode] = useState('interactive');
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  
  const { 
    connect, 
    disconnect, 
    sendAudio, 
    status,
    metrics 
  } = useVoiceSession();
  
  const {
    task,
    scope,
    time,
    mode,
    navigate
  } = use4DNavigation();

  useEffect(() => {
    if (waveformRef.current && !wavesurfer.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#6366f1',
        progressColor: '#4f46e5',
        cursorColor: '#818cf8',
        barWidth: 3,
        barRadius: 3,
        responsive: true,
        height: 60,
        normalize: true,
      });
    }
    return () => {
      wavesurfer.current?.destroy();
    };
  }, []);

  const handleMicToggle = async () => {
    if (!isListening) {
      await connect();
      setIsListening(true);
    } else {
      await disconnect();
      setIsListening(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto p-6">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-5xl font-bold text-white mb-2">
            4D Tesseract Voice AI
          </h1>
          <p className="text-purple-200">
            Navigate through Task • Scope • Time • Mode
          </p>
        </motion.div>

        {/* 4D Navigation Visualization */}
        <div className="grid grid-cols-4 gap-4 mb-8">
          <NavigationAxis
            icon={<Activity className="w-5 h-5" />}
            label="Task"
            value={task}
            options={['Command', 'Query', 'Navigate', 'Tool']}
            onChange={(v) => navigate('task', v)}
          />
          <NavigationAxis
            icon={<Layers className="w-5 h-5" />}
            label="Scope"
            value={scope}
            options={['Local', 'Project', 'System', 'Global']}
            onChange={(v) => navigate('scope', v)}
          />
          <NavigationAxis
            icon={<Clock className="w-5 h-5" />}
            label="Time"
            value={time}
            options={['Immediate', 'Short', 'Long', 'Recurring']}
            onChange={(v) => navigate('time', v)}
          />
          <NavigationAxis
            icon={<Grid3x3 className="w-5 h-5" />}
            label="Mode"
            value={mode}
            options={['Interactive', 'Batch', 'Analytical', 'Learning']}
            onChange={(v) => navigate('mode', v)}
          />
        </div>

        {/* Main Voice Interface */}
        <motion.div 
          className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 mb-8"
          whileHover={{ scale: 1.01 }}
        >
          <div className="flex flex-col items-center">
            {/* Waveform */}
            <div className="w-full mb-6">
              <div ref={waveformRef} className="rounded-lg" />
            </div>

            {/* Mic Button */}
            <motion.button
              onClick={handleMicToggle}
              className={`p-8 rounded-full transition-all ${
                isListening 
                  ? 'bg-red-500 hover:bg-red-600' 
                  : 'bg-indigo-500 hover:bg-indigo-600'
              }`}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <AnimatePresence mode="wait">
                {isListening ? (
                  <motion.div
                    key="listening"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  >
                    <MicOff className="w-12 h-12 text-white" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="not-listening"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  >
                    <Mic className="w-12 h-12 text-white" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.button>

            <p className="mt-4 text-white/80">
              {isListening ? 'Listening...' : 'Click to start'}
            </p>
          </div>
        </motion.div>

        {/* Module Status Grid */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          <ModuleStatus 
            name="Session Core" 
            status={status.sessionCore}
            latency={metrics.sessionLatency}
          />
          <ModuleStatus 
            name="ASR Engine" 
            status={status.asrEngine}
            latency={metrics.asrLatency}
          />
          <ModuleStatus 
            name="NLU Orchestrator" 
            status={status.nluOrchestrator}
            latency={metrics.nluLatency}
          />
          <ModuleStatus 
            name="Tool Plugins" 
            status={status.toolPlugins}
            latency={metrics.toolLatency}
          />
          <ModuleStatus 
            name="TTS Engine" 
            status={status.ttsEngine}
            latency={metrics.ttsLatency}
          />
          <ModuleStatus 
            name="Memory/Logs" 
            status={status.memoryLogs}
            latency={metrics.memoryLatency}
          />
        </div>

        {/* Performance Metrics */}
        <motion.div 
          className="bg-white/5 backdrop-blur rounded-xl p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex justify-between items-center">
            <div className="text-white/60">
              <span className="text-sm">Total Latency:</span>
              <span className={`ml-2 font-mono ${
                metrics.totalLatency < 300 ? 'text-green-400' : 'text-yellow-400'
              }`}>
                {metrics.totalLatency}ms
              </span>
            </div>
            <div className="text-white/60">
              <span className="text-sm">Session ID:</span>
              <span className="ml-2 font-mono text-purple-400">
                {status.sessionId || 'Disconnected'}
              </span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}