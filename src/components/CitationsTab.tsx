import React, { useState } from 'react';
import { Quote, Copy, Download, Plus, BookOpen, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import type { Citation } from '../types';
import toast from 'react-hot-toast';

export function CitationsTab() {
  const { state, dispatch } = useApp();
  const [selectedStyle, setSelectedStyle] = useState<'APA' | 'MLA' | 'Chicago' | 'Harvard'>('APA');
  const [isAdding, setIsAdding] = useState(false);
  const [newCitation, setNewCitation] = useState({
    title: '',
    authors: '',
    year: '',
    source: '',
    url: ''
  });

  const generateCitation = (citation: Citation, style: string) => {
    const authors = citation.authors.join(', ');
    const year = citation.year;
    const title = citation.title;
    const source = citation.source;

    switch (style) {
      case 'APA':
        return `${authors} (${year}). ${title}. ${source}.`;
      case 'MLA':
        return `${authors}. "${title}." ${source}, ${year}.`;
      case 'Chicago':
        return `${authors}. "${title}." ${source} (${year}).`;
      case 'Harvard':
        return `${authors} ${year}, '${title}', ${source}.`;
      default:
        return citation.text;
    }
  };

  const addCitation = () => {
    if (!newCitation.title || !newCitation.authors) return;

    const citation: Citation = {
      id: Date.now().toString(),
      type: selectedStyle,
      text: generateCitation({
        id: '',
        type: selectedStyle,
        text: '',
        source: newCitation.source,
        authors: newCitation.authors.split(',').map(a => a.trim()),
        year: newCitation.year,
        title: newCitation.title
      }, selectedStyle),
      source: newCitation.source,
      authors: newCitation.authors.split(',').map(a => a.trim()),
      year: newCitation.year,
      title: newCitation.title
    };

    dispatch({ type: 'ADD_CITATION', payload: citation });
    setNewCitation({ title: '', authors: '', year: '', source: '', url: '' });
    setIsAdding(false);
    toast.success('Citation added successfully!');
  };

  const copyCitation = (citation: Citation) => {
    const formattedCitation = generateCitation(citation, selectedStyle);
    navigator.clipboard.writeText(formattedCitation);
    toast.success('Citation copied to clipboard!');
  };

  const exportCitations = () => {
    const citations = state.citations.map(citation => 
      generateCitation(citation, selectedStyle)
    ).join('\n\n');
    
    const blob = new Blob([citations], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `citations-${selectedStyle.toLowerCase()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Citations exported successfully!');
  };

  // Generate sample citations from research papers
  const generateFromResearch = () => {
    state.researchPapers.forEach(paper => {
      const citation: Citation = {
        id: Date.now().toString() + Math.random(),
        type: selectedStyle,
        text: generateCitation({
          id: '',
          type: selectedStyle,
          text: '',
          source: paper.source,
          authors: paper.authors ? paper.authors.split(',').map(a => a.trim()) : ['Unknown'],
          year: paper.year,
          title: paper.title
        }, selectedStyle),
        source: paper.source,
        authors: paper.authors ? paper.authors.split(',').map(a => a.trim()) : ['Unknown'],
        year: paper.year,
        title: paper.title
      };

      dispatch({ type: 'ADD_CITATION', payload: citation });
    });
    toast.success(`Generated ${state.researchPapers.length} citations from research papers!`);
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
          <div className="w-10 h-10 bg-gradient-to-r from-slate-500 to-gray-500 rounded-lg flex items-center justify-center">
            <Quote className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Citations ({state.citations.length})
          </h2>
        </div>

        <div className="flex items-center space-x-4">
          {state.researchPapers.length > 0 && (
            <motion.button
              onClick={generateFromResearch}
              className="btn-outline inline-flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <BookOpen className="w-4 h-4" />
              <span>From Research</span>
            </motion.button>
          )}
          
          <motion.button
            onClick={() => setIsAdding(true)}
            className="btn-primary inline-flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="w-4 h-4" />
            <span>Add Citation</span>
          </motion.button>
        </div>
      </div>

      {/* Citation Style Selector */}
      <div className="flex items-center justify-between">
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {(['APA', 'MLA', 'Chicago', 'Harvard'] as const).map(style => (
            <button
              key={style}
              onClick={() => setSelectedStyle(style)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                selectedStyle === style
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-300'
              }`}
            >
              {style}
            </button>
          ))}
        </div>

        {state.citations.length > 0 && (
          <motion.button
            onClick={exportCitations}
            className="btn-outline inline-flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Download className="w-4 h-4" />
            <span>Export</span>
          </motion.button>
        )}
      </div>

      {/* Add Citation Modal */}
      <AnimatePresence>
        {isAdding && (
          <motion.div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div 
              className="bg-white dark:bg-gray-800 rounded-2xl p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
            >
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Add Citation</h3>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Title *
                  </label>
                  <input
                    type="text"
                    value={newCitation.title}
                    onChange={(e) => setNewCitation({ ...newCitation, title: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Enter title..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Authors * (comma separated)
                  </label>
                  <input
                    type="text"
                    value={newCitation.authors}
                    onChange={(e) => setNewCitation({ ...newCitation, authors: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Smith, J., Doe, A."
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Year
                    </label>
                    <input
                      type="text"
                      value={newCitation.year}
                      onChange={(e) => setNewCitation({ ...newCitation, year: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      placeholder="2024"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Source
                    </label>
                    <input
                      type="text"
                      value={newCitation.source}
                      onChange={(e) => setNewCitation({ ...newCitation, source: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      placeholder="Journal Name"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    URL (optional)
                  </label>
                  <input
                    type="url"
                    value={newCitation.url}
                    onChange={(e) => setNewCitation({ ...newCitation, url: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="https://..."
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-4 mt-8">
                <button
                  onClick={() => setIsAdding(false)}
                  className="px-6 py-3 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <motion.button
                  onClick={addCitation}
                  className="btn-primary"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Add Citation
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Citations List */}
      {state.citations.length === 0 ? (
        <motion.div 
          className="text-center py-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Quote className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Citations Yet
          </h3>
          <p className="text-gray-600 dark:text-gray-300">
            Add citations manually or generate them from your research papers.
          </p>
        </motion.div>
      ) : (
        <div className="space-y-6">
          {state.citations.map((citation, index) => (
            <motion.div
              key={citation.id}
              className="card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    {citation.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 text-sm mb-2">
                    {citation.authors.join(', ')} • {citation.year} • {citation.source}
                  </p>
                </div>
                
                <div className="flex space-x-2">
                  <motion.button
                    onClick={() => copyCitation(citation)}
                    className="p-2 text-gray-400 hover:text-blue-500 transition-colors"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Copy className="w-4 h-4" />
                  </motion.button>
                  {citation.source.startsWith('http') && (
                    <motion.a
                      href={citation.source}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-2 text-gray-400 hover:text-green-500 transition-colors"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <ExternalLink className="w-4 h-4" />
                    </motion.a>
                  )}
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <p className="text-gray-800 dark:text-gray-200 font-mono text-sm leading-relaxed">
                  {generateCitation(citation, selectedStyle)}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}