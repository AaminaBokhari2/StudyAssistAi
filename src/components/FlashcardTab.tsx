import React, { useEffect } from 'react';
import { Brain, ChevronLeft, ChevronRight, RotateCcw, Award, Lightbulb } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

export function FlashcardTab() {
  const { state, dispatch } = useApp();
  const { flashcards, flashcardState } = state;

  const currentCard = flashcards[flashcardState.currentCard];

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          handleFlip();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          handlePrevious();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleNext();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [flashcardState]);

  const handleFlip = () => {
    dispatch({
      type: 'UPDATE_FLASHCARD_STATE',
      payload: { isFlipped: !flashcardState.isFlipped }
    });
  };

  const handleNext = () => {
    if (flashcardState.currentCard < flashcards.length - 1) {
      dispatch({
        type: 'UPDATE_FLASHCARD_STATE',
        payload: {
          currentCard: flashcardState.currentCard + 1,
          isFlipped: false
        }
      });
    }
  };

  const handlePrevious = () => {
    if (flashcardState.currentCard > 0) {
      dispatch({
        type: 'UPDATE_FLASHCARD_STATE',
        payload: {
          currentCard: flashcardState.currentCard - 1,
          isFlipped: false
        }
      });
    }
  };

  const handleReset = () => {
    dispatch({
      type: 'UPDATE_FLASHCARD_STATE',
      payload: { currentCard: 0, isFlipped: false }
    });
  };

  if (state.isLoading && flashcards.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" text="Creating interactive flashcards" />
      </div>
    );
  }

  if (flashcards.length === 0) {
    return (
      <div className="text-center py-12">
        <Brain className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          No Flashcards Available
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to generate interactive flashcards.
        </p>
      </div>
    );
  }

  const getDifficultyColor = (difficulty?: string) => {
    switch (difficulty?.toLowerCase()) {
      case 'easy':
        return 'from-green-500 to-green-600';
      case 'medium':
        return 'from-yellow-500 to-yellow-600';
      case 'hard':
        return 'from-red-500 to-red-600';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  return (
    <motion.div 
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Progress and Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Brain className="w-6 h-6 text-primary-500" />
            <span className="text-xl font-bold text-gray-900 dark:text-white">
              Card {flashcardState.currentCard + 1} of {flashcards.length}
            </span>
          </div>
          
          {currentCard?.category && (
            <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-sm rounded-full font-medium">
              {currentCard.category}
            </span>
          )}
          
          {currentCard?.difficulty && (
            <span className={`px-3 py-1 text-white text-sm rounded-full font-medium bg-gradient-to-r ${getDifficultyColor(currentCard.difficulty)}`}>
              {currentCard.difficulty}
            </span>
          )}
        </div>

        <button
          onClick={handleReset}
          className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors btn-outline"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Reset</span>
        </button>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
        <motion.div
          className="bg-gradient-to-r from-primary-500 to-primary-600 h-3 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${((flashcardState.currentCard + 1) / flashcards.length) * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {/* Flashcard */}
      <div className="flex justify-center">
        <div className="w-full max-w-2xl">
          <motion.div
            className="flashcard"
            onClick={handleFlip}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <motion.div
              className="flashcard-inner"
              animate={{ rotateY: flashcardState.isFlipped ? 180 : 0 }}
              transition={{ duration: 0.6, type: "spring" }}
            >
              {/* Front */}
              <div className="flashcard-face bg-white dark:bg-gray-800 shadow-2xl">
                <div className="text-center">
                  <motion.div 
                    className="mb-6"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Brain className="w-8 h-8 text-white" />
                    </div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-wide font-medium">
                      Question
                    </p>
                  </motion.div>
                  <h3 className="text-2xl font-semibold text-gray-900 dark:text-white leading-relaxed mb-6">
                    {currentCard?.question}
                  </h3>
                  <motion.p 
                    className="text-sm text-gray-500 dark:text-gray-400 flex items-center justify-center space-x-2"
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <span>Click to reveal answer</span>
                    <Award className="w-4 h-4" />
                  </motion.p>
                </div>
              </div>

              {/* Back */}
              <div className="flashcard-face flashcard-back bg-gradient-to-br from-primary-500 to-primary-600 text-white shadow-2xl">
                <div className="text-center">
                  <motion.div 
                    className="mb-6"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    <div className="w-16 h-16 bg-white bg-opacity-20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Award className="w-8 h-8 text-white" />
                    </div>
                    <p className="text-sm text-primary-100 uppercase tracking-wide font-medium">
                      Answer
                    </p>
                  </motion.div>
                  <h3 className="text-2xl font-semibold leading-relaxed mb-6">
                    {currentCard?.answer}
                  </h3>
                  
                  <AnimatePresence>
                    {currentCard?.hint && (
                      <motion.div 
                        className="mt-6 p-4 bg-white bg-opacity-20 rounded-lg"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                      >
                        <div className="flex items-center justify-center space-x-2 mb-2">
                          <Lightbulb className="w-4 h-4" />
                          <span className="text-sm font-medium">Hint</span>
                        </div>
                        <p className="text-sm text-primary-100">
                          {currentCard.hint}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  
                  <p className="text-sm text-primary-100 mt-6">
                    Click to flip back
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-center space-x-6">
        <motion.button
          onClick={handlePrevious}
          disabled={flashcardState.currentCard === 0}
          className="flex items-center space-x-2 px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <ChevronLeft className="w-5 h-5" />
          <span>Previous</span>
        </motion.button>

        <div className="flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-primary-50 to-secondary-50 dark:from-primary-900/20 dark:to-secondary-900/20 text-primary-700 dark:text-primary-300 rounded-lg">
          <Brain className="w-5 h-5" />
          <span className="font-semibold">
            {flashcardState.currentCard + 1} / {flashcards.length}
          </span>
        </div>

        <motion.button
          onClick={handleNext}
          disabled={flashcardState.currentCard === flashcards.length - 1}
          className="flex items-center space-x-2 px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <span>Next</span>
          <ChevronRight className="w-5 h-5" />
        </motion.button>
      </div>

      {/* Keyboard Shortcuts */}
      <motion.div 
        className="text-center text-sm text-gray-500 dark:text-gray-400"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
      >
        <p className="flex items-center justify-center space-x-4">
          <span className="flex items-center space-x-1">
            <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">Space</kbd>
            <span>to flip</span>
          </span>
          <span className="flex items-center space-x-1">
            <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">←</kbd>
            <span>Previous</span>
          </span>
          <span className="flex items-center space-x-1">
            <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono">→</kbd>
            <span>Next</span>
          </span>
        </p>
      </motion.div>
    </motion.div>
  );
}