import React from 'react';
import { Video, ExternalLink, Clock, Eye, User, Play } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

export function VideosTab() {
  const { state } = useApp();
  const { youtubeVideos } = state;

  if (state.isLoading && youtubeVideos.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" text="Finding educational videos" />
      </div>
    );
  }

  if (youtubeVideos.length === 0) {
    return (
      <motion.div 
        className="text-center py-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="w-20 h-20 bg-gradient-to-r from-pink-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-6">
          <Video className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          No Videos Found
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to discover related educational videos.
        </p>
      </motion.div>
    );
  }

  const getEducationalScoreColor = (score?: string) => {
    switch (score?.toLowerCase()) {
      case 'high':
        return 'from-green-500 to-green-600';
      case 'medium':
        return 'from-yellow-500 to-yellow-600';
      case 'low':
        return 'from-red-500 to-red-600';
      default:
        return 'from-blue-500 to-blue-600';
    }
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
          <div className="w-10 h-10 bg-gradient-to-r from-pink-500 to-pink-600 rounded-lg flex items-center justify-center">
            <Video className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Educational Videos ({youtubeVideos.length} found)
          </h2>
        </div>
      </div>

      {/* Videos Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {youtubeVideos.map((video, index) => (
          <motion.div 
            key={index} 
            className="card card-hover overflow-hidden"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            {/* Video Thumbnail Placeholder */}
            <div className="aspect-video bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-700 dark:to-gray-800 flex items-center justify-center relative group">
              <div className="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
                <Play className="w-8 h-8 text-white ml-1" />
              </div>
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all duration-200" />
            </div>

            <div className="p-6">
              {/* Title */}
              <h3 className="font-bold text-gray-900 dark:text-white mb-3 line-clamp-2 leading-tight text-lg">
                {video.title}
              </h3>

              {/* Channel */}
              <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300 mb-4">
                <User className="w-4 h-4" />
                <span className="font-medium">{video.channel}</span>
              </div>

              {/* Metadata */}
              <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 mb-4">
                <div className="flex items-center space-x-4">
                  {video.duration && (
                    <div className="flex items-center space-x-1">
                      <Clock className="w-4 h-4" />
                      <span>{video.duration}</span>
                    </div>
                  )}
                  
                  {video.views && (
                    <div className="flex items-center space-x-1">
                      <Eye className="w-4 h-4" />
                      <span>{video.views}</span>
                    </div>
                  )}
                </div>

                {video.educational_score && (
                  <span className={`px-2 py-1 text-white text-xs rounded-full bg-gradient-to-r ${getEducationalScoreColor(video.educational_score)}`}>
                    {video.educational_score}
                  </span>
                )}
              </div>

              {/* Description */}
              {video.description && (
                <p className="text-sm text-gray-600 dark:text-gray-300 mb-6 line-clamp-3 leading-relaxed">
                  {video.description}
                </p>
              )}

              {/* Watch Button */}
              {video.url && video.url !== '#' && (
                <motion.a
                  href={video.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 w-full justify-center px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Play className="w-4 h-4" />
                  <span>Watch on YouTube</span>
                  <ExternalLink className="w-4 h-4" />
                </motion.a>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Educational Note */}
      <motion.div 
        className="card p-6 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-start space-x-4">
          <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <Video className="w-6 h-6 text-white" />
          </div>
          <div>
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 text-lg">
              Curated Educational Content
            </h4>
            <p className="text-blue-800 dark:text-blue-200 leading-relaxed">
              These videos are carefully selected from top educational channels like Khan Academy, Crash Course, MIT, TED-Ed, and other verified educational sources to provide high-quality learning content related to your document.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}