import React, { useState, useEffect } from 'react';
import { Timer, Play, Pause, RotateCcw, Coffee, Brain, Settings } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';

export function FocusTimerTab() {
  const { state, dispatch } = useApp();
  const { timerState } = state;
  const [customTime, setCustomTime] = useState(25);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (timerState.isRunning && timerState.timeLeft > 0) {
      interval = setInterval(() => {
        dispatch({
          type: 'UPDATE_TIMER_STATE',
          payload: { timeLeft: timerState.timeLeft - 1 }
        });
      }, 1000);
    } else if (timerState.timeLeft === 0 && timerState.isRunning) {
      // Timer finished
      dispatch({
        type: 'UPDATE_TIMER_STATE',
        payload: { 
          isRunning: false,
          cycles: timerState.mode === 'pomodoro' ? timerState.cycles + 1 : timerState.cycles
        }
      });
      
      // Play notification sound (if available)
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('Focus Timer', {
          body: timerState.mode === 'pomodoro' ? 'Time for a break!' : 'Break time over!',
          icon: '/vite.svg'
        });
      }
    }

    return () => clearInterval(interval);
  }, [timerState.isRunning, timerState.timeLeft, timerState.mode, timerState.cycles, dispatch]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startTimer = () => {
    dispatch({
      type: 'UPDATE_TIMER_STATE',
      payload: { isRunning: true }
    });
  };

  const pauseTimer = () => {
    dispatch({
      type: 'UPDATE_TIMER_STATE',
      payload: { isRunning: false }
    });
  };

  const resetTimer = () => {
    const defaultTime = timerState.mode === 'pomodoro' ? 25 * 60 : 
                      timerState.mode === 'break' ? 5 * 60 : 
                      customTime * 60;
    
    dispatch({
      type: 'UPDATE_TIMER_STATE',
      payload: { 
        isRunning: false,
        timeLeft: defaultTime
      }
    });
  };

  const setMode = (mode: 'pomodoro' | 'break' | 'custom') => {
    const timeMap = {
      pomodoro: 25 * 60,
      break: 5 * 60,
      custom: customTime * 60
    };

    dispatch({
      type: 'UPDATE_TIMER_STATE',
      payload: {
        mode,
        timeLeft: timeMap[mode],
        isRunning: false
      }
    });
  };

  const progress = timerState.mode === 'pomodoro' ? 
    ((25 * 60 - timerState.timeLeft) / (25 * 60)) * 100 :
    timerState.mode === 'break' ?
    ((5 * 60 - timerState.timeLeft) / (5 * 60)) * 100 :
    ((customTime * 60 - timerState.timeLeft) / (customTime * 60)) * 100;

  return (
    <motion.div 
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-r from-rose-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Timer className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Focus Timer
          </h2>
        </div>
      </div>

      {/* Timer Modes */}
      <div className="flex justify-center">
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-2xl p-2">
          <button
            onClick={() => setMode('pomodoro')}
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
              timerState.mode === 'pomodoro'
                ? 'bg-gradient-to-r from-rose-500 to-pink-500 text-white shadow-lg'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Brain className="w-4 h-4" />
            <span>Focus (25min)</span>
          </button>
          <button
            onClick={() => setMode('break')}
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
              timerState.mode === 'break'
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Coffee className="w-4 h-4" />
            <span>Break (5min)</span>
          </button>
          <button
            onClick={() => setMode('custom')}
            className={`px-6 py-3 rounded-xl text-sm font-medium transition-all duration-200 flex items-center space-x-2 ${
              timerState.mode === 'custom'
                ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Settings className="w-4 h-4" />
            <span>Custom</span>
          </button>
        </div>
      </div>

      {/* Custom Time Input */}
      <AnimatePresence>
        {timerState.mode === 'custom' && (
          <motion.div 
            className="flex justify-center"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="flex items-center space-x-4">
              <label className="text-gray-700 dark:text-gray-300 font-medium">
                Minutes:
              </label>
              <input
                type="number"
                min="1"
                max="120"
                value={customTime}
                onChange={(e) => setCustomTime(parseInt(e.target.value) || 1)}
                className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              />
              <button
                onClick={() => setMode('custom')}
                className="btn-primary px-4 py-2 text-sm"
              >
                Set
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Timer Display */}
      <div className="flex justify-center">
        <motion.div 
          className="relative"
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <svg className="w-80 h-80 transform -rotate-90" viewBox="0 0 100 100">
            {/* Background circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              className="text-gray-200 dark:text-gray-700"
            />
            {/* Progress circle */}
            <motion.circle
              cx="50"
              cy="50"
              r="45"
              stroke="url(#gradient)"
              strokeWidth="4"
              fill="none"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 45}`}
              strokeDashoffset={`${2 * Math.PI * 45 * (1 - progress / 100)}`}
              className="transition-all duration-1000 ease-out"
            />
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={
                  timerState.mode === 'pomodoro' ? '#f43f5e' :
                  timerState.mode === 'break' ? '#10b981' : '#3b82f6'
                } />
                <stop offset="100%" stopColor={
                  timerState.mode === 'pomodoro' ? '#ec4899' :
                  timerState.mode === 'break' ? '#059669' : '#8b5cf6'
                } />
              </linearGradient>
            </defs>
          </svg>
          
          {/* Timer text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.div 
              className="text-6xl font-bold text-gray-900 dark:text-white mb-2"
              animate={timerState.isRunning ? { scale: [1, 1.05, 1] } : {}}
              transition={{ duration: 1, repeat: timerState.isRunning ? Infinity : 0 }}
            >
              {formatTime(timerState.timeLeft)}
            </motion.div>
            <div className="text-lg text-gray-600 dark:text-gray-300 capitalize">
              {timerState.mode} Mode
            </div>
            {timerState.cycles > 0 && (
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Cycles completed: {timerState.cycles}
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Controls */}
      <div className="flex justify-center space-x-4">
        <motion.button
          onClick={timerState.isRunning ? pauseTimer : startTimer}
          className={`px-8 py-4 rounded-2xl font-semibold text-white shadow-lg transition-all duration-200 flex items-center space-x-3 ${
            timerState.isRunning
              ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700'
              : 'bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {timerState.isRunning ? (
            <>
              <Pause className="w-6 h-6" />
              <span>Pause</span>
            </>
          ) : (
            <>
              <Play className="w-6 h-6" />
              <span>Start</span>
            </>
          )}
        </motion.button>

        <motion.button
          onClick={resetTimer}
          className="px-8 py-4 rounded-2xl font-semibold text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 shadow-lg transition-all duration-200 flex items-center space-x-3"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw className="w-6 h-6" />
          <span>Reset</span>
        </motion.button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div 
          className="card p-6 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Timer className="w-6 h-6 text-white" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            {timerState.cycles}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            Pomodoros Today
          </div>
        </motion.div>

        <motion.div 
          className="card p-6 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            {Math.round(timerState.cycles * 25)}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            Minutes Focused
          </div>
        </motion.div>

        <motion.div 
          className="card p-6 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Coffee className="w-6 h-6 text-white" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            {Math.round(timerState.cycles * 0.2)}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            Breaks Taken
          </div>
        </motion.div>
      </div>

      {/* Tips */}
      <motion.div 
        className="card p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
          Pomodoro Technique Tips
        </h3>
        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
          <li>• Focus on one task during each 25-minute session</li>
          <li>• Take a 5-minute break after each pomodoro</li>
          <li>• Take a longer 15-30 minute break after 4 pomodoros</li>
          <li>• Eliminate distractions during focus time</li>
          <li>• Use breaks to rest your mind and body</li>
        </ul>
      </motion.div>
    </motion.div>
  );
}