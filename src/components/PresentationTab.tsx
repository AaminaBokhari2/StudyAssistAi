import React, { useState } from 'react';
import { Presentation, Play, Download, Settings, Palette, CheckCircle, AlertCircle, Star } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import { LoadingSpinner } from './LoadingSpinner';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

interface PresentationData {
  title: string;
  slides: Array<{
    title: string;
    content: string[];
    slide_type: string;
    notes: string;
  }>;
  theme: string;
  total_slides: number;
}

interface DesignGuidelines {
  color_scheme: string[];
  fonts: {
    heading: string;
    body: string;
  };
  layout: {
    style: string;
    spacing: string;
  };
  suggestions: string;
}

interface QualityAssessment {
  issues: string[];
  suggestions: string[];
  quality_assessment: string;
  overall_score: number;
  success: boolean;
}

export function PresentationTab() {
  const { state } = useApp();
  const [isGenerating, setIsGenerating] = useState(false);
  const [presentation, setPresentation] = useState<PresentationData | null>(null);
  const [designGuidelines, setDesignGuidelines] = useState<DesignGuidelines | null>(null);
  const [qualityAssessment, setQualityAssessment] = useState<QualityAssessment | null>(null);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [formData, setFormData] = useState({
    topic: '',
    audience: 'general',
    duration: 10,
    theme: 'professional'
  });

  const themes = [
    { id: 'professional', name: 'Professional', color: 'from-blue-600 to-indigo-600' },
    { id: 'creative', name: 'Creative', color: 'from-purple-600 to-pink-600' },
    { id: 'minimal', name: 'Minimal', color: 'from-gray-600 to-slate-600' },
    { id: 'modern', name: 'Modern', color: 'from-green-600 to-teal-600' }
  ];

  const audiences = [
    { id: 'general', name: 'General Audience' },
    { id: 'academic', name: 'Academic' },
    { id: 'business', name: 'Business' },
    { id: 'students', name: 'Students' },
    { id: 'technical', name: 'Technical' }
  ];

  const generatePresentation = async () => {
    if (!formData.topic.trim()) {
      toast.error('Please enter a presentation topic');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await apiService.generatePresentation({
        topic: formData.topic,
        audience: formData.audience,
        duration: formData.duration,
        theme: formData.theme
      });

      if (response.success) {
        setPresentation(response.presentation);
        setDesignGuidelines(response.design_guidelines);
        setQualityAssessment(response.quality_assessment);
        setCurrentSlide(0);
        toast.success('Presentation generated successfully!');
      } else {
        toast.error('Failed to generate presentation');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate presentation';
      toast.error(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  };

  const exportPresentation = () => {
    if (!presentation) return;

    const content = `# ${presentation.title}\n\n${presentation.slides.map((slide, index) => 
      `## Slide ${index + 1}: ${slide.title}\n\n${slide.content.join('\n\n')}\n\n**Notes:** ${slide.notes}\n\n---\n\n`
    ).join('')}`;

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${presentation.title.replace(/\s+/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Presentation exported successfully!');
  };

  if (!state.session.active) {
    return (
      <motion.div 
        className="text-center py-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="w-20 h-20 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
          <Presentation className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          Presentation Maker Ready
        </h3>
        <p className="text-gray-600 dark:text-gray-300">
          Upload a PDF document to create AI-powered presentations.
        </p>
      </motion.div>
    );
  }

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
          <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Presentation className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Presentation Maker
          </h2>
        </div>
      </div>

      {!presentation ? (
        /* Generation Form */
        <motion.div 
          className="card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
            Create New Presentation
          </h3>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Presentation Topic *
              </label>
              <input
                type="text"
                value={formData.topic}
                onChange={(e) => setFormData({ ...formData, topic: e.target.value })}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                placeholder="Enter your presentation topic..."
                disabled={isGenerating}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Target Audience
                </label>
                <select
                  value={formData.audience}
                  onChange={(e) => setFormData({ ...formData, audience: e.target.value })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  disabled={isGenerating}
                >
                  {audiences.map(audience => (
                    <option key={audience.id} value={audience.id}>
                      {audience.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Duration (minutes)
                </label>
                <input
                  type="number"
                  min="5"
                  max="60"
                  value={formData.duration}
                  onChange={(e) => setFormData({ ...formData, duration: parseInt(e.target.value) || 10 })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  disabled={isGenerating}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Theme
                </label>
                <select
                  value={formData.theme}
                  onChange={(e) => setFormData({ ...formData, theme: e.target.value })}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  disabled={isGenerating}
                >
                  {themes.map(theme => (
                    <option key={theme.id} value={theme.id}>
                      {theme.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex justify-center">
              <motion.button
                onClick={generatePresentation}
                disabled={isGenerating || !formData.topic.trim()}
                className="btn-primary inline-flex items-center space-x-3 px-8 py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isGenerating ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span>Generating Presentation...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Generate Presentation</span>
                  </>
                )}
              </motion.button>
            </div>
          </div>
        </motion.div>
      ) : (
        /* Presentation Display */
        <div className="space-y-8">
          {/* Presentation Header */}
          <div className="card p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  {presentation.title}
                </h3>
                <div className="flex items-center space-x-6 text-sm text-gray-600 dark:text-gray-300">
                  <span>{presentation.total_slides} slides</span>
                  <span>Theme: {presentation.theme}</span>
                  {qualityAssessment && (
                    <div className="flex items-center space-x-2">
                      <Star className="w-4 h-4 text-yellow-500" />
                      <span>Quality: {qualityAssessment.overall_score}/10</span>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <motion.button
                  onClick={exportPresentation}
                  className="btn-outline inline-flex items-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </motion.button>
                
                <motion.button
                  onClick={() => {
                    setPresentation(null);
                    setDesignGuidelines(null);
                    setQualityAssessment(null);
                  }}
                  className="btn-primary inline-flex items-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Settings className="w-4 h-4" />
                  <span>New Presentation</span>
                </motion.button>
              </div>
            </div>
          </div>

          {/* Slide Navigation */}
          <div className="flex justify-center">
            <div className="flex items-center space-x-4 bg-gray-100 dark:bg-gray-700 rounded-2xl p-2">
              <button
                onClick={() => setCurrentSlide(Math.max(0, currentSlide - 1))}
                disabled={currentSlide === 0}
                className="px-4 py-2 rounded-xl text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              
              <span className="px-4 py-2 bg-white dark:bg-gray-600 rounded-xl text-sm font-medium text-gray-900 dark:text-white">
                {currentSlide + 1} / {presentation.slides.length}
              </span>
              
              <button
                onClick={() => setCurrentSlide(Math.min(presentation.slides.length - 1, currentSlide + 1))}
                disabled={currentSlide === presentation.slides.length - 1}
                className="px-4 py-2 rounded-xl text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>

          {/* Current Slide */}
          <AnimatePresence mode="wait">
            <motion.div
              key={currentSlide}
              className="card p-12 min-h-[500px]"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-center space-y-8">
                <motion.h2 
                  className="text-4xl font-bold text-gray-900 dark:text-white mb-8"
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  {presentation.slides[currentSlide].title}
                </motion.h2>
                
                <motion.div 
                  className="space-y-6 text-left max-w-4xl mx-auto"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  {presentation.slides[currentSlide].content.map((item, index) => (
                    <motion.div
                      key={index}
                      className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + index * 0.1 }}
                    >
                      {item.startsWith('•') || item.startsWith('-') ? (
                        <div className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-primary-500 rounded-full mt-3 flex-shrink-0" />
                          <span>{item.replace(/^[•-]\s*/, '')}</span>
                        </div>
                      ) : (
                        <p>{item}</p>
                      )}
                    </motion.div>
                  ))}
                </motion.div>

                {presentation.slides[currentSlide].notes && (
                  <motion.div 
                    className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 }}
                  >
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                      Speaker Notes
                    </h4>
                    <p className="text-blue-800 dark:text-blue-200 text-sm">
                      {presentation.slides[currentSlide].notes}
                    </p>
                  </motion.div>
                )}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Design Guidelines & Quality Assessment */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Design Guidelines */}
            {designGuidelines && (
              <motion.div 
                className="card p-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <div className="flex items-center space-x-3 mb-4">
                  <Palette className="w-5 h-5 text-purple-500" />
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Design Guidelines
                  </h3>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Color Scheme</h4>
                    <div className="flex space-x-2">
                      {designGuidelines.color_scheme.map((color, index) => (
                        <div
                          key={index}
                          className="w-8 h-8 rounded-full border-2 border-gray-300"
                          style={{ backgroundColor: color }}
                          title={color}
                        />
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Typography</h4>
                    <div className="text-sm text-gray-600 dark:text-gray-300">
                      <p>Heading: {designGuidelines.fonts.heading}</p>
                      <p>Body: {designGuidelines.fonts.body}</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Layout</h4>
                    <div className="text-sm text-gray-600 dark:text-gray-300">
                      <p>Style: {designGuidelines.layout.style}</p>
                      <p>Spacing: {designGuidelines.layout.spacing}</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Suggestions</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      {designGuidelines.suggestions}
                    </p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Quality Assessment */}
            {qualityAssessment && (
              <motion.div 
                className="card p-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                <div className="flex items-center space-x-3 mb-4">
                  {qualityAssessment.success ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-yellow-500" />
                  )}
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Quality Assessment
                  </h3>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-500" />
                    <span className="text-sm font-medium">{qualityAssessment.overall_score}/10</span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                      {qualityAssessment.quality_assessment}
                    </p>
                  </div>
                  
                  {qualityAssessment.issues.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white mb-2">Issues</h4>
                      <ul className="space-y-1">
                        {qualityAssessment.issues.map((issue, index) => (
                          <li key={index} className="text-sm text-red-600 dark:text-red-400 flex items-start space-x-2">
                            <span className="w-1 h-1 bg-red-500 rounded-full mt-2 flex-shrink-0" />
                            <span>{issue}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {qualityAssessment.suggestions.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white mb-2">Suggestions</h4>
                      <ul className="space-y-1">
                        {qualityAssessment.suggestions.map((suggestion, index) => (
                          <li key={index} className="text-sm text-blue-600 dark:text-blue-400 flex items-start space-x-2">
                            <span className="w-1 h-1 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                            <span>{suggestion}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}