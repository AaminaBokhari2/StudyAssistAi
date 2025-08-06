import React from 'react';
import { BarChart3, TrendingUp, Clock, Target, Brain, Award } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';

export function AnalyticsTab() {
  const { state } = useApp();

  // Calculate analytics data
  const totalStudyTime = state.timerState.cycles * 25; // minutes
  const completedGoals = state.studyGoals.filter(goal => goal.completed).length;
  const averageQuizScore = state.quiz.length > 0 ? 
    Math.round((state.quizState.score / Math.max(Object.keys(state.quizState.answers).length, 1)) * 100) : 0;
  const notesCount = state.studyNotes.length;
  const flashcardsReviewed = state.flashcards.length > 0 ? state.flashcardState.currentCard + 1 : 0;

  const stats = [
    {
      title: 'Study Time',
      value: `${totalStudyTime}m`,
      change: '+12%',
      trend: 'up',
      icon: Clock,
      color: 'from-blue-500 to-blue-600',
      description: 'Total focused time'
    },
    {
      title: 'Quiz Performance',
      value: `${averageQuizScore}%`,
      change: '+8%',
      trend: 'up',
      icon: Brain,
      color: 'from-green-500 to-green-600',
      description: 'Average quiz score'
    },
    {
      title: 'Goals Completed',
      value: completedGoals,
      change: '+3',
      trend: 'up',
      icon: Target,
      color: 'from-purple-500 to-purple-600',
      description: 'Study goals achieved'
    },
    {
      title: 'Notes Created',
      value: notesCount,
      change: '+5',
      trend: 'up',
      icon: Award,
      color: 'from-orange-500 to-orange-600',
      description: 'Study notes written'
    }
  ];

  const weeklyData = [
    { day: 'Mon', minutes: 45, score: 85 },
    { day: 'Tue', minutes: 60, score: 92 },
    { day: 'Wed', minutes: 30, score: 78 },
    { day: 'Thu', minutes: 75, score: 88 },
    { day: 'Fri', minutes: 90, score: 95 },
    { day: 'Sat', minutes: 120, score: 90 },
    { day: 'Sun', minutes: 80, score: 87 }
  ];

  const maxMinutes = Math.max(...weeklyData.map(d => d.minutes));

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
          <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-green-500 rounded-lg flex items-center justify-center">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Study Analytics
          </h2>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            className="card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            whileHover={{ scale: 1.05, y: -5 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-12 h-12 bg-gradient-to-r ${stat.color} rounded-lg flex items-center justify-center`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <div className={`flex items-center space-x-1 text-sm ${
                stat.trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              }`}>
                <TrendingUp className="w-4 h-4" />
                <span>{stat.change}</span>
              </div>
            </div>
            
            <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
              {stat.value}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300 mb-2">
              {stat.title}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {stat.description}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Weekly Study Chart */}
      <motion.div 
        className="card p-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
          Weekly Study Activity
        </h3>
        
        <div className="space-y-6">
          {/* Chart */}
          <div className="flex items-end justify-between h-64 space-x-4">
            {weeklyData.map((data, index) => (
              <motion.div
                key={data.day}
                className="flex-1 flex flex-col items-center space-y-2"
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                transition={{ delay: 0.6 + index * 0.1 }}
              >
                <div className="w-full flex flex-col items-center space-y-1">
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {data.score}%
                  </div>
                  <motion.div
                    className="w-full bg-gradient-to-t from-blue-500 to-blue-400 rounded-t-lg"
                    style={{ height: `${(data.minutes / maxMinutes) * 200}px` }}
                    initial={{ height: 0 }}
                    animate={{ height: `${(data.minutes / maxMinutes) * 200}px` }}
                    transition={{ delay: 0.8 + index * 0.1, duration: 0.5 }}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {data.minutes}m
                  </div>
                </div>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {data.day}
                </div>
              </motion.div>
            ))}
          </div>
          
          {/* Legend */}
          <div className="flex items-center justify-center space-x-8 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-gradient-to-r from-blue-500 to-blue-400 rounded"></div>
              <span className="text-gray-600 dark:text-gray-300">Study Minutes</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-gray-400 rounded"></div>
              <span className="text-gray-600 dark:text-gray-300">Quiz Score %</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Study Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Progress Overview */}
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Study Progress
          </h3>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-600 dark:text-gray-300">Flashcards Reviewed</span>
                <span className="font-medium">{flashcardsReviewed}/{state.flashcards.length}</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                  style={{ width: `${state.flashcards.length > 0 ? (flashcardsReviewed / state.flashcards.length) * 100 : 0}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-600 dark:text-gray-300">Quiz Progress</span>
                <span className="font-medium">{Object.keys(state.quizState.answers).length}/{state.quiz.length}</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full"
                  style={{ width: `${state.quiz.length > 0 ? (Object.keys(state.quizState.answers).length / state.quiz.length) * 100 : 0}%` }}
                />
              </div>
            </div>
            
            {state.studyPlan && (
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600 dark:text-gray-300">Study Plan</span>
                  <span className="font-medium">{state.studyPlan.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-teal-500 to-cyan-500 h-2 rounded-full"
                    style={{ width: `${state.studyPlan.progress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Study Recommendations */}
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Recommendations
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Focus Time:</strong> You've studied {totalStudyTime} minutes this week. Try to reach 300 minutes for optimal learning.
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Quiz Performance:</strong> Your average score is {averageQuizScore}%. Review flashcards for topics you missed.
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Note Taking:</strong> You have {notesCount} notes. Consider organizing them by topic for better review.
                </p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
              <div>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Study Schedule:</strong> Maintain consistent daily study sessions for better retention.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}