import React, { useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import { AppProvider, useApp } from './contexts/AppContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Header } from './components/Header';
import { FileUpload } from './components/FileUpload';
import { StudyTabs } from './components/StudyTabs';
import { SummaryTab } from './components/SummaryTab';
import { FlashcardTab } from './components/FlashcardTab';
import { QuizTab } from './components/QuizTab';
import { QATab } from './components/QATab';
import { ResearchTab } from './components/ResearchTab';
import { VideosTab } from './components/VideosTab';
import { ResourcesTab } from './components/ResourcesTab';
import { StudyNotesTab } from './components/StudyNotesTab';
import { StudyPlannerTab } from './components/StudyPlannerTab';
import { FocusTimerTab } from './components/FocusTimerTab';
import { CitationsTab } from './components/CitationsTab';
import { AnalyticsTab } from './components/AnalyticsTab';
import { BookmarksTab } from './components/BookmarksTab';
import { PresentationTab } from './components/PresentationTab';
import { FullPageLoader } from './components/LoadingSpinner';
import { InteractiveBackground, FloatingShapes } from './components/InteractiveBackground';
import { motion, AnimatePresence } from 'framer-motion';

function AppContent() {
  const { state } = useApp();

  // Apply theme to document
  useEffect(() => {
    if (state.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [state.theme]);

  const renderTabContent = () => {
    switch (state.currentTab) {
      case 'summary':
        return <SummaryTab />;
      case 'flashcards':
        return <FlashcardTab />;
      case 'quiz':
        return <QuizTab />;
      case 'qa':
        return <QATab />;
      case 'notes':
        return <StudyNotesTab />;
      case 'planner':
        return <StudyPlannerTab />;
      case 'timer':
        return <FocusTimerTab />;
      case 'research':
        return <ResearchTab />;
      case 'videos':
        return <VideosTab />;
      case 'resources':
        return <ResourcesTab />;
      case 'citations':
        return <CitationsTab />;
      case 'concepts':
        return <div className="text-center py-12"><h3 className="text-xl font-semibold text-gray-900 dark:text-white">Mind Map - Coming Soon!</h3></div>;
      case 'analytics':
        return <AnalyticsTab />;
      case 'bookmarks':
        return <BookmarksTab />;
      case 'presentation':
        return <PresentationTab />;
      default:
        return <SummaryTab />;
    }
  };

  return (
    <div className="min-h-screen transition-colors duration-300 relative overflow-hidden particle-bg">
      {/* Interactive Background */}
      <InteractiveBackground />
      <FloatingShapes />
      
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {!state.session.active ? (
          <motion.div 
            className="max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div 
              className="text-center mb-16 relative"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.2 }}
            >
              <div className="absolute inset-0 flex items-center justify-center">
                <motion.div 
                  className="w-32 h-32 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full blur-3xl"
                  animate={{ 
                    scale: [1, 1.2, 1],
                    rotate: [0, 180, 360]
                  }}
                  transition={{ 
                    duration: 8,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              </div>
              <motion.h1 
                className="text-6xl md:text-7xl font-bold gradient-text mb-6 relative z-10 tracking-tight text-shadow-glow"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
              >
                AI Study Assistant
              </motion.h1>
              <motion.div 
                className="inline-block px-6 py-2 bg-gradient-to-r from-blue-500/10 to-purple-500/10 backdrop-blur-sm rounded-full border border-blue-200/30 dark:border-purple-300/30 mb-4 neon-border"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <p className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Professional Edition
                </p>
              </motion.div>
              <motion.p 
                className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.8 }}
              >
                Transform your documents into comprehensive study materials
              </motion.p>
              <motion.p 
                className="text-lg text-gray-500 dark:text-gray-400 max-w-2xl mx-auto leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 1.0 }}
              >
                Create flashcards, quizzes, summaries, and discover related research—all powered by advanced AI technology with beautiful, interactive interfaces.
              </motion.p>
            </motion.div>
            <FileUpload />
          </motion.div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div 
              className="space-y-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
            <StudyTabs />
            <motion.div 
              className="relative"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.2 }}
            >
              {renderTabContent()}
            </motion.div>
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      <AnimatePresence>
        {state.isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <FullPageLoader text="Processing your document with AI" />
          </motion.div>
        )}
      </AnimatePresence>
      
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: state.theme === 'dark' ? '#374151' : '#ffffff',
            color: state.theme === 'dark' ? '#f9fafb' : '#111827',
            border: `1px solid ${state.theme === 'dark' ? '#4b5563' : '#e5e7eb'}`,
            borderRadius: '12px',
            padding: '16px',
            fontSize: '14px',
            fontWeight: '500',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#ffffff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#ffffff',
            },
          },
        }}
      />
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <AppProvider>
        <AppContent />
      </AppProvider>
    </ErrorBoundary>
  );
}

export default App;