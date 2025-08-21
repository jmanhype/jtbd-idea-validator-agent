import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface NavigationAxisProps {
  icon: ReactNode;
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}

export function NavigationAxis({ icon, label, value, options, onChange }: NavigationAxisProps) {
  return (
    <motion.div 
      className="bg-white/10 backdrop-blur rounded-xl p-4"
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="text-purple-400">{icon}</div>
        <span className="text-white font-medium">{label}</span>
      </div>
      
      <div className="space-y-2">
        {options.map((option) => (
          <button
            key={option}
            onClick={() => onChange(option.toLowerCase())}
            className={`w-full px-3 py-2 rounded-lg text-sm transition-all ${
              value === option.toLowerCase()
                ? 'bg-purple-500/50 text-white border border-purple-400'
                : 'bg-white/5 text-white/60 hover:bg-white/10 border border-transparent'
            }`}
          >
            {option}
          </button>
        ))}
      </div>
      
      <div className="mt-3 pt-3 border-t border-white/10">
        <div className="text-xs text-white/40">Current:</div>
        <div className="text-sm font-mono text-purple-300">{value}</div>
      </div>
    </motion.div>
  );
}