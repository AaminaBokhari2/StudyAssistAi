import React, { useState } from 'react';
import { Calendar, Clock, Target, CheckCircle, Plus, Edit, Trash2, Play } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import type { StudyPlan, StudyTopic, StudyGoal } from '../types';

export function StudyPlannerTab() {
  const { state, dispatch } = useApp();
  const [activeView, setActiveView] = useState<'plan' | 'goals'>('plan');
  const [isCreatingPlan, setIsCreatingPlan] = useState(false);
  const [isCreatingGoal, setIsCreatingGoal] = useState(false);

  const generateStudyPlan = () => {
    const topics: StudyTopic[] = [
      {
        id: '1',
        title: 'Document Overview',
        description: 'Read and understand the main concepts',
        estimatedTime: '30 minutes',
        completed: false,
        resources: ['Summary', 'Key Points'],
        notes: ''
      },
      {
        id: '2',
        title: 'Active Learning',
        description: 'Practice with flashcards and quizzes',
        estimatedTime: '45 minutes',
        completed: false,
        resources: ['Flashcards', 'Quiz'],
        notes: ''
      },
      {
        id: '3',
        title: 'Research & Exploration',
        description: 'Explore related research and resources',
        estimatedTime: '60 minutes',
        completed: false,
        resources: ['Research Papers', 'Videos', 'Web Resources'],
        notes: ''
      },
      {
        id: '4',
        title: 'Review & Consolidation',
        description: 'Review notes and test understanding',
        estimatedTime: '30 minutes',
        completed: false,
        resources: ['Notes', 'Q&A'],
        notes: ''
      }
    ];

    const plan: StudyPlan = {
      id: Date.now().toString(),
      title: 'Comprehensive Study Plan',
      description: 'AI-generated study plan based on your document',
      duration: '2-3 hours',
      difficulty: 'Intermediate',
      topics,
      progress: 0,
      createdAt: new Date()
    };

    dispatch({ type: 'SET_STUDY_PLAN', payload: plan });
  };

  const toggleTopicCompletion = (topicId: string) => {
    if (!state.studyPlan) return;

    const updatedTopics = state.studyPlan.topics.map(topic => 
      topic.id === topicId ? { ...topic, completed: !topic.completed } : topic
    );

    const completedCount = updatedTopics.filter(topic => topic.completed).length;
    const progress = Math.round((completedCount / updatedTopics.length) * 100);

    dispatch({ 
      type: 'UPDATE_STUDY_PLAN', 
      payload: { topics: updatedTopics, progress }
    });
  };

  const addGoal = (goalData: Omit<StudyGoal, 'id'>) => {
    const goal: StudyGoal = {
      ...goalData,
      id: Date.now().toString()
    };
    dispatch({ type: 'ADD_STUDY_GOAL', payload: goal });
  };

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
          <div className="w-10 h-10 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-lg flex items-center justify-center">
            <Calendar className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Study Planner
          </h2>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setActiveView('plan')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'plan'
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-300'
              }`}
            >
              Study Plan
            </button>
            <button
              onClick={() => setActiveView('goals')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'goals'
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-300'
              }`}
            >
              Goals
            </button>
          </div>
        </div>
      </div>

      {activeView === 'plan' && (
        <div className="space-y-6">
          {!state.studyPlan ? (
            <motion.div 
              className="text-center py-12"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <Calendar className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                No Study Plan Yet
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Generate a personalized study plan based on your document.
              </p>
              <motion.button
                onClick={generateStudyPlan}
                className="btn-primary inline-flex items-center space-x-2"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Plus className="w-4 h-4" />
                <span>Generate Study Plan</span>
              </motion.button>
            </motion.div>
          ) : (
            <div className="space-y-6">
              {/* Plan Overview */}
              <div className="card p-8">
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      {state.studyPlan.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300 mb-4">
                      {state.studyPlan.description}
                    </p>
                    <div className="flex items-center space-x-6 text-sm">
                      <div className="flex items-center space-x-2">
                        <Clock className="w-4 h-4 text-blue-500" />
                        <span className="text-gray-600 dark:text-gray-300">
                          {state.studyPlan.duration}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Target className="w-4 h-4 text-green-500" />
                        <span className="text-gray-600 dark:text-gray-300">
                          {state.studyPlan.difficulty}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-3xl font-bold text-primary-600 mb-1">
                      {state.studyPlan.progress}%
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      Complete
                    </div>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-6">
                  <motion.div
                    className="bg-gradient-to-r from-teal-500 to-cyan-500 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${state.studyPlan.progress}%` }}
                    transition={{ duration: 1 }}
                  />
                </div>
              </div>

              {/* Topics */}
              <div className="space-y-4">
                {state.studyPlan.topics.map((topic, index) => (
                  <motion.div
                    key={topic.id}
                    className={`card p-6 ${topic.completed ? 'bg-green-50 dark:bg-green-900/20' : ''}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <div className="flex items-start space-x-4">
                      <motion.button
                        onClick={() => toggleTopicCompletion(topic.id)}
                        className={`w-6 h-6 rounded-full border-2 flex items-center justify-center mt-1 ${
                          topic.completed
                            ? 'bg-green-500 border-green-500 text-white'
                            : 'border-gray-300 dark:border-gray-600 hover:border-green-500'
                        }`}
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                      >
                        {topic.completed && <CheckCircle className="w-4 h-4" />}
                      </motion.button>

                      <div className="flex-1">
                        <h4 className={`font-semibold mb-2 ${
                          topic.completed 
                            ? 'text-green-700 dark:text-green-300 line-through' 
                            : 'text-gray-900 dark:text-white'
                        }`}>
                          {topic.title}
                        </h4>
                        <p className="text-gray-600 dark:text-gray-300 mb-3">
                          {topic.description}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 text-sm">
                            <div className="flex items-center space-x-1">
                              <Clock className="w-4 h-4 text-blue-500" />
                              <span className="text-gray-500 dark:text-gray-400">
                                {topic.estimatedTime}
                              </span>
                            </div>
                          </div>
                          
                          <div className="flex flex-wrap gap-2">
                            {topic.resources.map(resource => (
                              <span
                                key={resource}
                                className="px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-full"
                              >
                                {resource}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeView === 'goals' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              Study Goals ({state.studyGoals.length})
            </h3>
            <motion.button
              onClick={() => setIsCreatingGoal(true)}
              className="btn-primary inline-flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Plus className="w-4 h-4" />
              <span>Add Goal</span>
            </motion.button>
          </div>

          {state.studyGoals.length === 0 ? (
            <div className="text-center py-12">
              <Target className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No Goals Set
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Set study goals to track your progress and stay motivated.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {state.studyGoals.map((goal, index) => (
                <motion.div
                  key={goal.id}
                  className="card p-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <div className="flex items-start justify-between mb-4">
                    <h4 className="font-semibold text-gray-900 dark:text-white">
                      {goal.title}
                    </h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      goal.priority === 'High' 
                        ? 'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300'
                        : goal.priority === 'Medium'
                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300'
                        : 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300'
                    }`}>
                      {goal.priority}
                    </span>
                  </div>
                  
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    {goal.description}
                  </p>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500 dark:text-gray-400">Progress</span>
                      <span className="font-medium">{goal.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-teal-500 to-cyan-500 h-2 rounded-full"
                        style={{ width: `${goal.progress}%` }}
                      />
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      Due: {goal.targetDate.toLocaleDateString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}