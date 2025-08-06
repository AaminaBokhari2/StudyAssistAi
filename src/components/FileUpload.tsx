import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, Brain, Search, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { apiService } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
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
            relative border-2 border-dashed rounded-3xl p-16 text-center cursor-pointer transition-all duration-500 backdrop-blur-xl
            ${isDragActive 
              ? 'border-blue-400 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 scale-105 shadow-2xl glow-effect' 
              : 'border-gray-300/50 dark:border-gray-600/50 hover:border-blue-400 hover:bg-gradient-to-br hover:from-blue-50/50 hover:to-purple-50/50 dark:hover:from-blue-900/10 dark:hover:to-purple-900/10 hover:shadow-xl glass-effect'
            }
            ${isUploading ? 'pointer-events-none' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          <div className="space-y-8">
            <motion.div 
              className="flex justify-center"
              animate={uploadStatus === 'processing' ? { rotate: 360 } : {}}
              transition={{ duration: 2, repeat: uploadStatus === 'processing' ? Infinity : 0, ease: "linear" }}
            >
              {getStatusIcon()}
            </motion.div>
            
            <div>
              <h3 className="text-3xl font-bold gradient-text mb-4">
                {getStatusText()}
              </h3>
              <p className="text-lg text-gray-600 dark:text-gray-300 leading-relaxed">
                {uploadStatus === 'idle' && (
                  <>
                    Drag and drop your PDF file here, or click to browse
                    <br />
                    <span className="text-base text-gray-500 dark:text-gray-400 font-medium">Maximum file size: 50MB</span>
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
                <p className="text-base text-gray-600 dark:text-gray-300 font-semibold">
                  {uploadProgress}% complete
                </p>
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
        <div className="card p-8 text-center card-hover">
          <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-6 glow-effect">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-bold text-xl text-gray-900 dark:text-white mb-3">Smart Analysis</h4>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">AI-powered document analysis with intelligent summaries and key insights</p>
        </div>
        
        <div className="card p-8 text-center card-hover">
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-6 glow-effect">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-bold text-xl text-gray-900 dark:text-white mb-3">Study Materials</h4>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">Interactive flashcards, adaptive quizzes, and personalized Q&A assistance</p>
        </div>
        
        <div className="card p-8 text-center card-hover">
          <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-6 glow-effect">
            <Search className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-bold text-xl text-gray-900 dark:text-white mb-3">Smart Discovery</h4>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">Research papers, educational videos, and curated learning resources</p>
        </div>
      </motion.div>

      <motion.div 
        className="mt-8 text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="inline-flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-full border border-blue-200/30 dark:border-purple-300/30 backdrop-blur-sm">
          <Zap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <span className="text-base font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Powered by Advanced AI Technology
          </span>
        </div>
      </motion.div>
    </div>
  );
}