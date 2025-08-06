import React from 'react';
import { Globe, ExternalLink, Star, BookOpen, Code, GraduationCap, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

export function ResourcesTab() {
  const { state } = useApp();
  const { webResources } = state;

  if (state.isLoading && webResources.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" text="Finding web resources" />
      </div>
    );
  }

  if (webResources.length === 0) {
    return (
      <motion.div 
        className="text-center py-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="w-20 h-20 bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-6">
          <Globe className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          No Web Resources Found
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to discover related web learning resources.
        </p>
      </motion.div>
    );
  }

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'course':
        return <GraduationCap className="w-5 h-5" />;
      case 'tutorial':
        return <Code className="w-5 h-5" />;
      case 'documentation':
        return <BookOpen className="w-5 h-5" />;
      default:
        return <Globe className="w-5 h-5" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'course':
        return 'from-green-500 to-green-600';
      case 'tutorial':
        return 'from-blue-500 to-blue-600';
      case 'documentation':
        return 'from-purple-500 to-purple-600';
      case 'reference':
        return 'from-orange-500 to-orange-600';
      default:
        return 'from-gray-500 to-gray-600';
    }
  };

  const getQualityScoreColor = (score?: string) => {
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
          <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-lg flex items-center justify-center">
            <Globe className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Web Learning Resources ({webResources.length} found)
          </h2>
        </div>
      </div>

      {/* Resources Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {webResources.map((resource, index) => (
          <motion.div 
            key={index} 
            className="card card-hover p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className={`p-3 rounded-lg bg-gradient-to-r ${getTypeColor(resource.type)}`}>
                  {getTypeIcon(resource.type)}
                </div>
                <span className={`px-3 py-1 text-white text-sm rounded-full bg-gradient-to-r ${getTypeColor(resource.type)}`}>
                  {resource.type}
                </span>
              </div>
              
              {resource.quality_score && (
                <span className={`px-3 py-1 text-white text-sm rounded-full bg-gradient-to-r ${getQualityScoreColor(resource.quality_score)}`}>
                  {resource.quality_score}
                </span>
              )}
            </div>

            {/* Title */}
            <h3 className="font-bold text-gray-900 dark:text-white mb-3 line-clamp-2 leading-tight text-lg">
              {resource.title}
            </h3>

            {/* Source */}
            <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300 mb-4">
              <Globe className="w-4 h-4" />
              <span className="font-medium">{resource.source}</span>
            </div>

            {/* Description */}
            {resource.description && (
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-6 line-clamp-3 leading-relaxed">
                {resource.description}
              </p>
            )}

            {/* Action Button */}
            {resource.url && resource.url !== '#' && (
              <motion.a
                href={resource.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center space-x-2 w-full justify-center px-6 py-3 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white font-medium rounded-lg transition-all duration-200"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Zap className="w-4 h-4" />
                <span>Visit Resource</span>
                <ExternalLink className="w-4 h-4" />
              </motion.a>
            )}
          </motion.div>
        ))}
      </div>

      {/* Platform Info */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="card p-6 text-center">
          <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <GraduationCap className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Courses</h4>
          <p className="text-sm text-gray-600 dark:text-gray-300">Coursera, edX, Udemy</p>
        </div>
        
        <div className="card p-6 text-center">
          <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Code className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Tutorials</h4>
          <p className="text-sm text-gray-600 dark:text-gray-300">W3Schools, MDN, FreeCodeCamp</p>
        </div>
        
        <div className="card p-6 text-center">
          <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <BookOpen className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Documentation</h4>
          <p className="text-sm text-gray-600 dark:text-gray-300">Official docs & guides</p>
        </div>
        
        <div className="card p-6 text-center">
          <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Star className="w-6 h-6 text-white" />
          </div>
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Quality</h4>
          <p className="text-sm text-gray-600 dark:text-gray-300">Verified platforms only</p>
        </div>
      </motion.div>
    </motion.div>
  );
}