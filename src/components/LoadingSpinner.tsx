import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

export function LoadingSpinner({ size = 'md', text, className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div className={`flex items-center justify-center space-x-2 ${className}`}>
      <Loader2 className={`animate-spin text-primary-500 ${sizeClasses[size]}`} />
      {text && (
        <span className="text-gray-600 dark:text-gray-300 animate-pulse">
          {text}
          <span className="loading-dots"></span>
        </span>
      )}
    </div>
  );
}

export function FullPageLoader({ text = 'Loading' }: { text?: string }) {
  return (
    <div className="fixed inset-0 bg-white dark:bg-gray-900 bg-opacity-80 dark:bg-opacity-80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full flex items-center justify-center mb-4 animate-bounce-gentle">
          <Loader2 className="w-8 h-8 text-white animate-spin" />
        </div>
        <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
          {text}
          <span className="loading-dots"></span>
        </p>
      </div>
    </div>
  );
}