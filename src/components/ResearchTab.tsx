import React from 'react';
import { BookOpen, ExternalLink, Star, Calendar, Users, Award } from 'lucide-react';
import { motion } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';

export function ResearchTab() {
  const { state } = useApp();
  const { researchPapers } = state;

  if (state.isLoading && researchPapers.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" text="Discovering research papers" />
      </div>
    );
  }

  if (researchPapers.length === 0) {
    return (
      <motion.div 
        className="text-center py-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="w-20 h-20 bg-gradient-to-r from-red-500 to-red-600 rounded-full flex items-center justify-center mx-auto mb-6">
          <BookOpen className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          No Research Papers Found
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to discover related research papers.
        </p>
      </motion.div>
    );
  }

  const getRelevanceColor = (score?: number, label?: string) => {
    if (label) {
      switch (label.toLowerCase()) {
        case 'high':
          return 'from-green-500 to-green-600';
        case 'medium':
          return 'from-yellow-500 to-yellow-600';
        case 'low':
          return 'from-red-500 to-red-600';
        default:
          return 'from-gray-500 to-gray-600';
      }
    }
    
    if (score) {
      if (score >= 0.8) return 'from-green-500 to-green-600';
      if (score >= 0.6) return 'from-yellow-500 to-yellow-600';
      return 'from-red-500 to-red-600';
    }
    
    return 'from-gray-500 to-gray-600';
  };

  const getRelevanceText = (score?: number, label?: string) => {
    if (label) return label;
    if (score) return `${Math.round(score * 100)}% match`;
    return 'N/A';
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
          <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-red-600 rounded-lg flex items-center justify-center">
            <BookOpen className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Research Papers ({researchPapers.length} found)
          </h2>
        </div>
      </div>

      {/* Papers Grid */}
      <div className="grid gap-8">
        {researchPapers.map((paper, index) => (
          <motion.div 
            key={index} 
            className="card card-hover p-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <div className="flex items-start justify-between mb-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white leading-tight flex-1 mr-6">
                {paper.title}
              </h3>
              
              {(paper.relevance_score || paper.relevance_label) && (
                <span className={`px-3 py-1 text-white text-sm rounded-full flex-shrink-0 bg-gradient-to-r ${getRelevanceColor(paper.relevance_score, paper.relevance_label)}`}>
                  {getRelevanceText(paper.relevance_score, paper.relevance_label)}
                </span>
              )}
            </div>

            {/* Metadata */}
            <div className="flex flex-wrap items-center gap-6 text-sm text-gray-600 dark:text-gray-300 mb-6">
              {paper.authors && (
                <div className="flex items-center space-x-2">
                  <Users className="w-4 h-4" />
                  <span>{paper.authors}</span>
                </div>
              )}
              
              {paper.year && (
                <div className="flex items-center space-x-2">
                  <Calendar className="w-4 h-4" />
                  <span>{paper.year}</span>
                </div>
              )}
              
              {paper.citation_count && (
                <div className="flex items-center space-x-2">
                  <Star className="w-4 h-4" />
                  <span>{paper.citation_count} citations</span>
                </div>
              )}
              
              {paper.source && (
                <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-sm rounded-full">
                  {paper.source}
                </span>
              )}
            </div>

            {/* Fields of Study */}
            {paper.fields_of_study && paper.fields_of_study.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-6">
                {paper.fields_of_study.slice(0, 4).map((field, fieldIndex) => (
                  <span
                    key={fieldIndex}
                    className="px-3 py-1 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 text-sm rounded-full"
                  >
                    {field}
                  </span>
                ))}
                {paper.fields_of_study.length > 4 && (
                  <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-sm rounded-full">
                    +{paper.fields_of_study.length - 4} more
                  </span>
                )}
              </div>
            )}

            {/* Abstract */}
            {paper.abstract && (
              <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6 text-lg">
                {paper.abstract}
              </p>
            )}

            {/* Actions */}
            <div className="flex items-center justify-between pt-6 border-t border-gray-200 dark:border-gray-700">
              <div className="flex flex-wrap gap-2">
                {paper.categories && paper.categories.slice(0, 3).map((category, catIndex) => (
                  <span
                    key={catIndex}
                    className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-xs rounded"
                  >
                    {category}
                  </span>
                ))}
              </div>
              
              {paper.url && paper.url !== '#' && (
                <motion.a
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white font-medium rounded-lg transition-all duration-200"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Award className="w-4 h-4" />
                  <span>Read Paper</span>
                  <ExternalLink className="w-4 h-4" />
                </motion.a>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}