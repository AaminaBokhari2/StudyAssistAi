import React from 'react';
import { FileText, Clock, BookOpen, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

export function SummaryTab() {
  const { state } = useApp();

  if (state.isLoading && !state.summary) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" text="Generating intelligent summary" />
      </div>
    );
  }

  if (!state.summary) {
    return (
      <div className="text-center py-12">
        <FileText className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          No Summary Available
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to generate an AI-powered summary.
        </p>
      </div>
    );
  }

  const readingTime = state.session.word_count ? Math.ceil(state.session.word_count / 200) : 0;

  return (
    <motion.div 
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-300">Words</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {state.session.word_count?.toLocaleString() || 'N/A'}
              </p>
            </div>
          </div>
        </motion.div>
        
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-lg flex items-center justify-center">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-300">Pages</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {state.session.page_count || 'N/A'}
              </p>
            </div>
          </div>
        </motion.div>
        
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.3 }}
        >
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Clock className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-300">Read Time</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {readingTime ? `${readingTime} min` : 'N/A'}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.4 }}
        >
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-300">Complexity</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">
                {readingTime > 10 ? 'High' : readingTime > 5 ? 'Medium' : 'Low'}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Summary Content */}
      <motion.div 
        className="card p-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
            <FileText className="w-4 h-4 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Intelligent Summary
          </h2>
        </div>
        
        <div className="prose prose-lg prose-gray dark:prose-invert max-w-none">
          <ReactMarkdown className="text-gray-700 dark:text-gray-300 leading-relaxed">
            {state.summary}
          </ReactMarkdown>
        </div>
      </motion.div>

      {/* Processing Methods */}
      {state.session.methods_used && state.session.methods_used.length > 0 && (
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Processing Methods Used
          </h3>
          <div className="flex flex-wrap gap-3">
            {state.session.methods_used.map((method, index) => (
              <motion.span
                key={index}
                className="px-3 py-2 bg-gradient-to-r from-primary-50 to-secondary-50 dark:from-primary-900/20 dark:to-secondary-900/20 text-primary-700 dark:text-primary-300 text-sm rounded-full border border-primary-200 dark:border-primary-700"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: 0.6 + index * 0.1 }}
              >
                {method}
              </motion.span>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}