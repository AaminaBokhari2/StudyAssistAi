import React from 'react';
import { GraduationCap, Moon, Sun, FileText, Zap, RotateCcw } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { apiService } from '../services/api';
import { AnimatedIcon, GlowingIcon } from './AnimatedIcon';
import toast from 'react-hot-toast';

export function Header() {
  const { state, dispatch } = useApp();

  const toggleTheme = () => {
    dispatch({ type: 'TOGGLE_THEME' });
  };

  const handleNewSession = async () => {
    try {
      await apiService.clearSession();
      dispatch({ type: 'CLEAR_SESSION' });
      toast.success('Session cleared successfully');
    } catch (error) {
      toast.error('Failed to clear session');
    }
  };

  return (
    <motion.header 
      className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl shadow-xl border-b border-gray-200/50 dark:border-gray-700/50 sticky top-0 z-40"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, type: "spring", stiffness: 100 }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo and Title */}
          <motion.div 
            className="flex items-center space-x-3"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <motion.div 
              className="w-12 h-12 bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg pulse-glow"
              animate={{ 
                rotate: [0, 5, -5, 0],
                scale: [1, 1.05, 1]
              }}
              transition={{ 
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <GraduationCap className="w-7 h-7 text-white" />
            </motion.div>
            <div>
              <motion.h1 
                className="text-2xl font-bold gradient-text"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                AI Study Assistant
              </motion.h1>
              <motion.p 
                className="text-sm text-blue-600 dark:text-blue-400 font-medium"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                Professional Edition
              </motion.p>
            </div>
          </motion.div>

          {/* Session Info */}
          {state.session.active && (
            <motion.div 
              className="hidden md:flex items-center space-x-6 text-sm"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <motion.div 
                className="flex items-center space-x-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-full"
                whileHover={{ scale: 1.05 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <AnimatedIcon icon={FileText} animation="float" size={16} />
                <span className="font-semibold text-blue-700 dark:text-blue-300">{state.session.word_count?.toLocaleString()} words</span>
              </motion.div>
              <motion.div 
                className="flex items-center space-x-2 px-4 py-2 bg-purple-50 dark:bg-purple-900/20 rounded-full"
                whileHover={{ scale: 1.05 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <AnimatedIcon icon={Zap} animation="pulse" size={16} />
                <span className="font-semibold text-purple-700 dark:text-purple-300">{state.session.page_count} pages</span>
              </motion.div>
            </motion.div>
          )}

          {/* Controls */}
          <div className="flex items-center space-x-2">
            {state.session.active && (
              <motion.button
                onClick={handleNewSession}
                className="p-3 rounded-xl bg-white/60 dark:bg-gray-700/60 backdrop-blur-sm hover:bg-white/80 dark:hover:bg-gray-600/80 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 border border-gray-200/50 dark:border-gray-600/50"
                title="New Session"
                whileHover={{ scale: 1.1, rotate: 180 }}
                whileTap={{ scale: 0.9 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <RotateCcw className="w-5 h-5 text-gray-700 dark:text-gray-300" />
              </motion.button>
            )}
            
            <motion.button
              onClick={toggleTheme}
              className="p-3 rounded-xl bg-white/60 dark:bg-gray-700/60 backdrop-blur-sm hover:bg-white/80 dark:hover:bg-gray-600/80 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 border border-gray-200/50 dark:border-gray-600/50"
              title="Toggle Theme"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <motion.div
                animate={{ rotate: state.theme === 'light' ? 0 : 180 }}
                transition={{ duration: 0.5 }}
              >
                {state.theme === 'light' ? (
                  <GlowingIcon icon={Moon} size={20} glowColor="#6366f1" />
                ) : (
                  <GlowingIcon icon={Sun} size={20} glowColor="#f59e0b" />
                )}
              </motion.div>
            </motion.button>
          </div>
        </div>
      </div>
    </motion.header>
  );
}