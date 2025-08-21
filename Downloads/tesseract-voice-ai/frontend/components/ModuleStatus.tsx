import { motion } from 'framer-motion';
import { Check, AlertCircle, Loader2 } from 'lucide-react';

interface ModuleStatusProps {
  name: string;
  status: string;
  latency?: number;
}

export function ModuleStatus({ name, status, latency }: ModuleStatusProps) {
  const getStatusIcon = () => {
    switch (status) {
      case 'ready':
      case 'connected':
        return <Check className="w-4 h-4 text-green-400" />;
      case 'processing':
      case 'executing':
      case 'synthesizing':
      case 'writing':
        return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <div className="w-4 h-4 rounded-full bg-gray-400/50" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'ready':
      case 'connected':
        return 'border-green-400/50 bg-green-400/10';
      case 'processing':
      case 'executing':
      case 'synthesizing':
      case 'writing':
        return 'border-blue-400/50 bg-blue-400/10';
      case 'error':
        return 'border-red-400/50 bg-red-400/10';
      default:
        return 'border-gray-400/50 bg-gray-400/10';
    }
  };

  return (
    <motion.div
      className={`border rounded-lg p-4 backdrop-blur ${getStatusColor()}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-white font-medium text-sm">{name}</h3>
        {getStatusIcon()}
      </div>
      
      <div className="flex items-center justify-between">
        <span className="text-xs text-white/60 capitalize">{status}</span>
        {latency !== undefined && (
          <span className={`text-xs font-mono ${
            latency < 50 ? 'text-green-400' : 
            latency < 100 ? 'text-yellow-400' : 
            'text-red-400'
          }`}>
            {latency}ms
          </span>
        )}
      </div>
    </motion.div>
  );
}