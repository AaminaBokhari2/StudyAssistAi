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
import { FullPageLoader } from './components/LoadingSpinner';

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
      case 'research':
        return <ResearchTab />;
      case 'videos':
        return <VideosTab />;
      case 'resources':
        return <ResourcesTab />;
      default:
        return <SummaryTab />;
    }
  };

  return (
    <div className="min-h-screen transition-colors duration-300 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-400/20 rounded-full blur-3xl floating-animation"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-indigo-400/20 to-pink-400/20 rounded-full blur-3xl floating-animation" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-purple-400/10 to-blue-400/10 rounded-full blur-3xl floating-animation" style={{ animationDelay: '4s' }}></div>
      </div>
      
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!state.session.active ? (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-16 relative">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-32 h-32 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full blur-3xl"></div>
              </div>
              <h1 className="text-6xl md:text-7xl font-bold gradient-text mb-6 relative z-10 tracking-tight">
                AI Study Assistant
              </h1>
              <div className="inline-block px-6 py-2 bg-gradient-to-r from-blue-500/10 to-purple-500/10 backdrop-blur-sm rounded-full border border-blue-200/30 dark:border-purple-300/30 mb-4">
                <p className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Professional Edition
                </p>
              </div>
              <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
                Professional Edition
              </p>
              <p className="text-lg text-gray-500 dark:text-gray-400 max-w-2xl mx-auto leading-relaxed">
                Transform your documents into comprehensive study materials with cutting-edge AI technology. Create flashcards, quizzes, summaries, and discover related research—all powered by advanced machine learning.
              </p>
            </div>
            <FileUpload />
          </div>
        ) : (
          <div className="space-y-8">
            <StudyTabs />
            <div className="animate-fade-in relative">
              {renderTabContent()}
            </div>
          </div>
        )}
      </main>

      {state.isLoading && <FullPageLoader text="Processing your document with AI" />}
      
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