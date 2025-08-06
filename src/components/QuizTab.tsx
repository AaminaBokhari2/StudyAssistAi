import React from 'react';
import { HelpCircle, ChevronLeft, ChevronRight, CheckCircle, XCircle, Trophy, RotateCcw, Target } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

export function QuizTab() {
  const { state, dispatch } = useApp();
  const { quiz, quizState } = state;

  const currentQuestion = quiz[quizState.currentQuestion];

  const handleAnswerSelect = (selectedAnswer: string) => {
    const newAnswers = { ...quizState.answers };
    newAnswers[quizState.currentQuestion] = selectedAnswer;
    
    dispatch({
      type: 'UPDATE_QUIZ_STATE',
      payload: { 
        answers: newAnswers,
        showExplanation: false
      }
    });
  };

  const handleSubmitAnswer = () => {
    const selectedAnswer = quizState.answers[quizState.currentQuestion];
    if (!selectedAnswer) return;

    const isCorrect = selectedAnswer === currentQuestion.options[currentQuestion.correct_answer];
    const newScore = isCorrect ? quizState.score + 1 : quizState.score;

    dispatch({
      type: 'UPDATE_QUIZ_STATE',
      payload: {
        score: newScore,
        showExplanation: true
      }
    });
  };

  const handleNext = () => {
    if (quizState.currentQuestion < quiz.length - 1) {
      dispatch({
        type: 'UPDATE_QUIZ_STATE',
        payload: {
          currentQuestion: quizState.currentQuestion + 1,
          showExplanation: false
        }
      });
    } else {
      // Complete quiz
      dispatch({
        type: 'UPDATE_QUIZ_STATE',
        payload: { completed: true }
      });
    }
  };

  const handlePrevious = () => {
    if (quizState.currentQuestion > 0) {
      dispatch({
        type: 'UPDATE_QUIZ_STATE',
        payload: {
          currentQuestion: quizState.currentQuestion - 1,
          showExplanation: false
        }
      });
    }
  };

  const handleReset = () => {
    dispatch({
      type: 'UPDATE_QUIZ_STATE',
      payload: {
        currentQuestion: 0,
        answers: {},
        score: 0,
        completed: false,
        showExplanation: false
      }
    });
  };

  if (state.isLoading && quiz.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" text="Creating adaptive quiz" />
      </div>
    );
  }

  if (quiz.length === 0) {
    return (
      <div className="text-center py-12">
        <HelpCircle className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          No Quiz Available
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to generate an adaptive quiz.
        </p>
      </div>
    );
  }

  if (quizState.completed) {
    const scorePercentage = Math.round((quizState.score / quiz.length) * 100);
    const getScoreInfo = () => {
      if (scorePercentage >= 90) return { message: "Excellent work!", icon: "🌟", color: "text-green-600", bgColor: "from-green-500 to-green-600" };
      if (scorePercentage >= 70) return { message: "Good job!", icon: "👍", color: "text-blue-600", bgColor: "from-blue-500 to-blue-600" };
      if (scorePercentage >= 50) return { message: "Keep studying!", icon: "📚", color: "text-yellow-600", bgColor: "from-yellow-500 to-yellow-600" };
      return { message: "Don't give up!", icon: "💪", color: "text-red-600", bgColor: "from-red-500 to-red-600" };
    };

    const scoreInfo = getScoreInfo();

    return (
      <motion.div 
        className="text-center py-12"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="max-w-md mx-auto">
          <motion.div
            className={`w-24 h-24 bg-gradient-to-r ${scoreInfo.bgColor} rounded-full flex items-center justify-center mx-auto mb-8`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", bounce: 0.5 }}
          >
            <Trophy className="w-12 h-12 text-white" />
          </motion.div>
          
          <motion.h2 
            className="text-3xl font-bold text-gray-900 dark:text-white mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            Quiz Completed!
          </motion.h2>
          
          <motion.div 
            className="card p-8 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="text-5xl font-bold text-primary-600 mb-2">
              {quizState.score}/{quiz.length}
            </div>
            <div className="text-3xl font-semibold text-gray-700 dark:text-gray-300 mb-4">
              {scorePercentage}%
            </div>
            <div className={`text-xl ${scoreInfo.color} font-medium`}>
              {scoreInfo.icon} {scoreInfo.message}
            </div>
          </motion.div>

          <motion.button
            onClick={handleReset}
            className="btn-primary inline-flex items-center space-x-2 text-lg px-8 py-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <RotateCcw className="w-5 h-5" />
            <span>Retake Quiz</span>
          </motion.button>
        </div>
      </motion.div>
    );
  }

  const selectedAnswer = quizState.answers[quizState.currentQuestion];
  const isAnswered = selectedAnswer !== undefined;
  const isCorrect = isAnswered && selectedAnswer === currentQuestion.options[currentQuestion.correct_answer];
  const answeredCount = Object.keys(quizState.answers).length;
  const scorePercentage = answeredCount > 0 ? Math.round((quizState.score / answeredCount) * 100) : 0;

  return (
    <motion.div 
      className="space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Progress and Score */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <HelpCircle className="w-6 h-6 text-primary-500" />
            <span className="text-xl font-bold text-gray-900 dark:text-white">
              Question {quizState.currentQuestion + 1} of {quiz.length}
            </span>
          </div>
          
          <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-primary-50 to-secondary-50 dark:from-primary-900/20 dark:to-secondary-900/20 rounded-full">
            <Trophy className="w-4 h-4 text-primary-600 dark:text-primary-400" />
            <span className="text-sm font-semibold text-primary-700 dark:text-primary-300">
              Score: {quizState.score}/{answeredCount} ({scorePercentage}%)
            </span>
          </div>
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
          animate={{ width: `${((quizState.currentQuestion + 1) / quiz.length) * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {/* Question */}
      <motion.div 
        className="card p-8"
        key={quizState.currentQuestion}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-start space-x-4 mb-8">
          <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <Target className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white leading-relaxed">
            {currentQuestion?.question}
          </h3>
        </div>

        {/* Options */}
        <div className="space-y-4">
          {currentQuestion?.options.map((option, index) => {
            const isSelected = selectedAnswer === option;
            const isCorrectOption = index === currentQuestion.correct_answer;
            
            let optionClass = "w-full text-left p-6 rounded-xl border-2 transition-all duration-200 ";
            
            if (quizState.showExplanation) {
              if (isCorrectOption) {
                optionClass += "border-green-500 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300";
              } else if (isSelected && !isCorrectOption) {
                optionClass += "border-red-500 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300";
              } else {
                optionClass += "border-gray-200 dark:border-gray-600 text-gray-700 dark:text-gray-300";
              }
            } else {
              if (isSelected) {
                optionClass += "border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300";
              } else {
                optionClass += "border-gray-200 dark:border-gray-600 hover:border-primary-300 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300";
              }
            }

            return (
              <motion.button
                key={index}
                onClick={() => !quizState.showExplanation && handleAnswerSelect(option)}
                disabled={quizState.showExplanation}
                className={optionClass}
                whileHover={!quizState.showExplanation ? { scale: 1.02 } : {}}
                whileTap={!quizState.showExplanation ? { scale: 0.98 } : {}}
              >
                <div className="flex items-center justify-between">
                  <span className="text-lg">{option}</span>
                  <AnimatePresence>
                    {quizState.showExplanation && (
                      <motion.div 
                        className="flex items-center space-x-2"
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0 }}
                      >
                        {isCorrectOption && <CheckCircle className="w-6 h-6 text-green-500" />}
                        {isSelected && !isCorrectOption && <XCircle className="w-6 h-6 text-red-500" />}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.button>
            );
          })}
        </div>

        {/* Submit Button */}
        <AnimatePresence>
          {!quizState.showExplanation && (
            <motion.div 
              className="mt-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <button
                onClick={handleSubmitAnswer}
                disabled={!isAnswered}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed text-lg px-8 py-4"
              >
                Submit Answer
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Explanation */}
        <AnimatePresence>
          {quizState.showExplanation && currentQuestion.explanation && (
            <motion.div 
              className="mt-8 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-3 flex items-center space-x-2">
                <HelpCircle className="w-5 h-5" />
                <span>Explanation</span>
              </h4>
              <p className="text-blue-800 dark:text-blue-200 leading-relaxed">
                {currentQuestion.explanation}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <motion.button
          onClick={handlePrevious}
          disabled={quizState.currentQuestion === 0}
          className="flex items-center space-x-2 px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <ChevronLeft className="w-5 h-5" />
          <span>Previous</span>
        </motion.button>

        <AnimatePresence>
          {quizState.showExplanation && (
            <motion.button
              onClick={handleNext}
              className="flex items-center space-x-2 btn-primary text-lg px-8 py-4"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span>{quizState.currentQuestion === quiz.length - 1 ? 'Finish Quiz' : 'Next Question'}</span>
              <ChevronRight className="w-5 h-5" />
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}