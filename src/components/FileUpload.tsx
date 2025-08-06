import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, Brain, Search, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { apiService } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { InteractiveCard } from './InteractiveCard';
import { AnimatedIcon } from './AnimatedIcon';
import toast from 'react-hot-toast';

export function FileUpload() {
  const { dispatch } = useApp();
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file
    if (file.type !== 'application/pdf') {
      toast.error('Please upload a PDF file only');
      return;
    }

    if (file.size > 50 * 1024 * 1024) { // 50MB limit
      toast.error('File size must be less than 50MB');
      return;
    }

    setIsUploading(true);
    setUploadStatus('uploading');
    setUploadProgress(0);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Upload and process PDF
      setUploadStatus('processing');
      const result = await apiService.uploadPDF(file);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadStatus('success');

      // Update session
      const sessionInfo = await apiService.getSessionInfo();
      dispatch({ type: 'SET_SESSION', payload: sessionInfo });

      // Generate all study materials
      await generateAllMaterials();

      toast.success('Document processed successfully!');
      
    } catch (error: any) {
      setUploadStatus('error');
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      toast.error(errorMessage);
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
      setTimeout(() => {
        setUploadStatus('idle');
        setUploadProgress(0);
      }, 2000);
    }
  }, [dispatch]);

  const generateAllMaterials = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    
    try {
      // Generate summary
      const summaryResult = await apiService.generateSummary();
      dispatch({ type: 'SET_SUMMARY', payload: summaryResult.summary });

      // Generate flashcards
      const flashcardsResult = await apiService.generateFlashcards();
      dispatch({ type: 'SET_FLASHCARDS', payload: flashcardsResult.flashcards });

      // Generate quiz
      const quizResult = await apiService.generateQuiz();
      dispatch({ type: 'SET_QUIZ', payload: quizResult.quiz });

      // Discover research papers
      const researchResult = await apiService.discoverResearch();
      dispatch({ type: 'SET_RESEARCH_PAPERS', payload: researchResult.papers });

      // Discover YouTube videos
      const videosResult = await apiService.discoverVideos();
      dispatch({ type: 'SET_YOUTUBE_VIDEOS', payload: videosResult.videos });

      // Discover web resources
      const resourcesResult = await apiService.discoverResources();
      dispatch({ type: 'SET_WEB_RESOURCES', payload: resourcesResult.resources });

    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate materials';
      toast.error(errorMessage);
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: false,
    disabled: isUploading
  });

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case 'uploading':
      case 'processing':
        return <LoadingSpinner size="lg" />;
      case 'success':
        return <CheckCircle className="w-12 h-12 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-12 h-12 text-red-500" />;
      default:
        return <Upload className="w-12 h-12 text-primary-500" />;
    }
  };

  const getStatusText = () => {
    switch (uploadStatus) {
      case 'uploading':
        return 'Uploading document...';
      case 'processing':
        return 'Processing with AI...';
      case 'success':
        return 'Successfully processed!';
      case 'error':
        return 'Upload failed';
      default:
        return isDragActive ? 'Drop your PDF here' : 'Upload your PDF document';
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-3xl p-16 text-center cursor-pointer transition-all duration-500 backdrop-blur-xl hover-lift
            ${isDragActive 
              ? 'border-blue-500 bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-800/40 dark:to-purple-800/40 scale-105 shadow-2xl pulse-glow' 
              : 'border-blue-300 dark:border-blue-500 bg-gradient-to-br from-white/90 to-blue-50/90 dark:from-gray-800/90 dark:to-blue-900/30 hover:border-blue-500 hover:from-blue-50 hover:to-purple-50 dark:hover:from-blue-900/40 dark:hover:to-purple-900/40 hover:shadow-xl hover:scale-105'
            }
            ${isUploading ? 'pointer-events-none' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          <div className="space-y-8">
            <motion.div
              className="flex justify-center"
              animate={uploadStatus === 'processing' ? { 
                rotate: 360,
                scale: [1, 1.1, 1]
              } : {}}
              transition={{ 
                rotate: { duration: 2, repeat: uploadStatus === 'processing' ? Infinity : 0, ease: "linear" },
                scale: { duration: 1, repeat: uploadStatus === 'processing' ? Infinity : 0 }
              }}
            >
              <div className="relative">
                {getStatusIcon()}
                {uploadStatus === 'idle' && (
                  <motion.div
                    className="absolute inset-0 rounded-full"
                    animate={{
                      boxShadow: [
                        '0 0 0 0 rgba(59, 130, 246, 0.4)',
                        '0 0 0 20px rgba(59, 130, 246, 0)',
                      ]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeOut"
                    }}
                  />
                )}
              </div>
            </motion.div>
            
            <div>
              <motion.h3 
                className="text-3xl font-bold text-gray-800 dark:text-white mb-4"
                animate={isDragActive ? { scale: 1.05 } : { scale: 1 }}
                transition={{ duration: 0.2 }}
              >
                {getStatusText()}
              </motion.h3>
              <p className="text-lg text-gray-700 dark:text-gray-200 leading-relaxed font-medium">
                {uploadStatus === 'idle' && (
                  <>
                    Drag and drop your PDF file here, or click to browse
                    <br />
                    <motion.span 
                      className="text-base text-gray-600 dark:text-gray-300 font-semibold"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      Maximum file size: 50MB
                    </motion.span>
                  </>
                )}
              </p>
            </div>

            {isUploading && (
              <motion.div 
                className="space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                  <motion.div 
                    className="progress-bar h-4"
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                <motion.p 
                  className="text-base text-gray-800 dark:text-gray-200 font-bold"
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                >
                  {uploadProgress}% complete
                </motion.p>
              </motion.div>
            )}
          </div>
        </div>
      </motion.div>

      <motion.div 
        className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <InteractiveCard
          title="Smart Analysis"
          description="AI-powered document analysis with intelligent summaries and key insights"
          icon={FileText}
          gradient="from-emerald-500 to-teal-500"
        />
        
        <InteractiveCard
          title="Study Materials"
          description="Interactive flashcards, adaptive quizzes, and personalized Q&A assistance"
          icon={Brain}
          gradient="from-purple-500 to-pink-500"
        />
        
        <InteractiveCard
          title="Smart Discovery"
          description="Research papers, educational videos, and curated learning resources"
          icon={Search}
          gradient="from-orange-500 to-red-500"
        />
      </motion.div>

      <motion.div 
        className="mt-8 text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <motion.div 
          className="inline-flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-full border border-blue-200/30 dark:border-purple-300/30 backdrop-blur-sm"
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          <AnimatedIcon icon={Zap} animation="pulse" size={20} color="#3b82f6" />
          <span className="text-base font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Powered by Advanced AI Technology
          </span>
        </motion.div>
      </motion.div>
    </div>
  );
}