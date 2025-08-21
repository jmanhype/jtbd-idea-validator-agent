import { useState, useCallback, useRef, useEffect } from 'react';
import SimplePeer from 'simple-peer';

interface VoiceSessionStatus {
  sessionCore: 'connected' | 'disconnected' | 'error';
  asrEngine: 'ready' | 'processing' | 'idle';
  nluOrchestrator: 'ready' | 'processing' | 'idle';
  toolPlugins: 'ready' | 'executing' | 'idle';
  ttsEngine: 'ready' | 'synthesizing' | 'idle';
  memoryLogs: 'ready' | 'writing' | 'idle';
  sessionId: string | null;
}

interface VoiceSessionMetrics {
  sessionLatency: number;
  asrLatency: number;
  nluLatency: number;
  toolLatency: number;
  ttsLatency: number;
  memoryLatency: number;
  totalLatency: number;
}

export function useVoiceSession() {
  const [status, setStatus] = useState<VoiceSessionStatus>({
    sessionCore: 'disconnected',
    asrEngine: 'idle',
    nluOrchestrator: 'idle',
    toolPlugins: 'idle',
    ttsEngine: 'idle',
    memoryLogs: 'idle',
    sessionId: null,
  });

  const [metrics, setMetrics] = useState<VoiceSessionMetrics>({
    sessionLatency: 0,
    asrLatency: 0,
    nluLatency: 0,
    toolLatency: 0,
    ttsLatency: 0,
    memoryLatency: 0,
    totalLatency: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const peerRef = useRef<SimplePeer.Instance | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const connect = useCallback(async () => {
    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        } 
      });
      streamRef.current = stream;

      // Connect WebSocket
      const ws = new WebSocket('ws://localhost:8000/ws/voice');
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus(prev => ({ ...prev, sessionCore: 'connected' }));
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'session_created') {
          setStatus(prev => ({ ...prev, sessionId: data.sessionId }));
        } else if (data.type === 'metrics') {
          setMetrics(data.metrics);
        } else if (data.type === 'status') {
          setStatus(prev => ({ ...prev, ...data.status }));
        } else if (data.type === 'offer') {
          handleWebRTCOffer(data.offer);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus(prev => ({ ...prev, sessionCore: 'error' }));
      };

      ws.onclose = () => {
        setStatus(prev => ({ ...prev, sessionCore: 'disconnected', sessionId: null }));
      };

      // Initialize WebRTC
      initializeWebRTC(stream);

    } catch (error) {
      console.error('Failed to connect:', error);
      setStatus(prev => ({ ...prev, sessionCore: 'error' }));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (peerRef.current) {
      peerRef.current.destroy();
      peerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setStatus({
      sessionCore: 'disconnected',
      asrEngine: 'idle',
      nluOrchestrator: 'idle',
      toolPlugins: 'idle',
      ttsEngine: 'idle',
      memoryLogs: 'idle',
      sessionId: null,
    });
  }, []);

  const sendAudio = useCallback((audioData: ArrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(audioData);
    }
  }, []);

  const initializeWebRTC = (stream: MediaStream) => {
    const peer = new SimplePeer({
      initiator: true,
      stream: stream,
      trickle: false,
      config: {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' },
        ],
      },
    });

    peer.on('signal', (data) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'signal', signal: data }));
      }
    });

    peer.on('stream', (remoteStream) => {
      // Handle remote audio stream (TTS output)
      const audio = new Audio();
      audio.srcObject = remoteStream;
      audio.play();
    });

    peer.on('error', (err) => {
      console.error('WebRTC error:', err);
    });

    peerRef.current = peer;
  };

  const handleWebRTCOffer = (offer: any) => {
    if (peerRef.current) {
      peerRef.current.signal(offer);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    sendAudio,
    status,
    metrics,
  };
}