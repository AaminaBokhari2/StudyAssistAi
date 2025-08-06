import React from 'react';
import { GraduationCap, Moon, Sun, FileText, Zap, RotateCcw } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { apiService } from '../services/api';
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
    <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl shadow-xl border-b border-gray-200/50 dark:border-gray-700/50 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg glow-effect">
              <GraduationCap className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">
                AI Study Assistant
              </h1>
              <p className="text-sm text-blue-600 dark:text-blue-400 font-medium">
                Professional Edition
              </p>
            </div>
          </div>

          {/* Session Info */}
          {state.session.active && (
            <div className="hidden md:flex items-center space-x-6 text-sm">
              <div className="flex items-center space-x-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-full">
                <FileText className="w-4 h-4" />
                <span className="font-semibold text-blue-700 dark:text-blue-300">{state.session.word_count?.toLocaleString()} words</span>
              </div>
              <div className="flex items-center space-x-2 px-4 py-2 bg-purple-50 dark:bg-purple-900/20 rounded-full">
                <Zap className="w-4 h-4" />
                <span className="font-semibold text-purple-700 dark:text-purple-300">{state.session.page_count} pages</span>
              </div>
            </div>
          )}

          {/* Controls */}
          <div className="flex items-center space-x-2">
            {state.session.active && (
              <button
                onClick={handleNewSession}
                className="p-3 rounded-xl bg-white/60 dark:bg-gray-700/60 backdrop-blur-sm hover:bg-white/80 dark:hover:bg-gray-600/80 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 border border-gray-200/50 dark:border-gray-600/50"
                title="New Session"
              >
                <RotateCcw className="w-5 h-5 text-gray-700 dark:text-gray-300" />
              </button>
            )}
            
            <button
              onClick={toggleTheme}
              className="p-3 rounded-xl bg-white/60 dark:bg-gray-700/60 backdrop-blur-sm hover:bg-white/80 dark:hover:bg-gray-600/80 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 border border-gray-200/50 dark:border-gray-600/50"
              title="Toggle Theme"
            >
              {state.theme === 'light' ? (
                <Moon className="w-5 h-5 text-gray-700 dark:text-gray-300" />
              ) : (
                <Sun className="w-5 h-5 text-gray-700 dark:text-gray-300" />
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}