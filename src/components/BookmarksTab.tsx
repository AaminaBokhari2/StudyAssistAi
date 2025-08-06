import React, { useState } from 'react';
import { Bookmark, Plus, ExternalLink, Search, Filter, Tag, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '../contexts/AppContext';
import type { BookmarkItem } from '../types';
import toast from 'react-hot-toast';

export function BookmarksTab() {
  const { state, dispatch } = useApp();
  const [isAdding, setIsAdding] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [newBookmark, setNewBookmark] = useState({
    title: '',
    url: '',
    description: '',
    tags: '',
    category: 'Research'
  });

  const categories = ['Research', 'Videos', 'Articles', 'Tools', 'References', 'Other'];

  const filteredBookmarks = state.bookmarks.filter(bookmark => {
    const matchesSearch = bookmark.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         bookmark.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         bookmark.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || bookmark.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const addBookmark = () => {
    if (!newBookmark.title || !newBookmark.url) return;

    const bookmark: BookmarkItem = {
      id: Date.now().toString(),
      title: newBookmark.title,
      url: newBookmark.url,
      description: newBookmark.description,
      tags: newBookmark.tags.split(',').map(tag => tag.trim()).filter(Boolean),
      category: newBookmark.category,
      addedAt: new Date()
    };

    dispatch({ type: 'ADD_BOOKMARK', payload: bookmark });
    setNewBookmark({ title: '', url: '', description: '', tags: '', category: 'Research' });
    setIsAdding(false);
    toast.success('Bookmark added successfully!');
  };

  const importFromResources = () => {
    // Import from research papers
    state.researchPapers.forEach(paper => {
      if (paper.url && paper.url !== '#') {
        const bookmark: BookmarkItem = {
          id: Date.now().toString() + Math.random(),
          title: paper.title,
          url: paper.url,
          description: paper.abstract || 'Research paper',
          tags: paper.fields_of_study || [],
          category: 'Research',
          addedAt: new Date()
        };
        dispatch({ type: 'ADD_BOOKMARK', payload: bookmark });
      }
    });

    // Import from videos
    state.youtubeVideos.forEach(video => {
      if (video.url && video.url !== '#') {
        const bookmark: BookmarkItem = {
          id: Date.now().toString() + Math.random(),
          title: video.title,
          url: video.url,
          description: video.description || 'Educational video',
          tags: ['video', 'education'],
          category: 'Videos',
          addedAt: new Date()
        };
        dispatch({ type: 'ADD_BOOKMARK', payload: bookmark });
      }
    });

    // Import from web resources
    state.webResources.forEach(resource => {
      if (resource.url && resource.url !== '#') {
        const bookmark: BookmarkItem = {
          id: Date.now().toString() + Math.random(),
          title: resource.title,
          url: resource.url,
          description: resource.description || 'Web resource',
          tags: [resource.type.toLowerCase()],
          category: 'Articles',
          addedAt: new Date()
        };
        dispatch({ type: 'ADD_BOOKMARK', payload: bookmark });
      }
    });

    toast.success('Imported bookmarks from your research resources!');
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
          <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
            <Bookmark className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Bookmarks ({filteredBookmarks.length})
          </h2>
        </div>

        <div className="flex items-center space-x-4">
          {(state.researchPapers.length > 0 || state.youtubeVideos.length > 0 || state.webResources.length > 0) && (
            <motion.button
              onClick={importFromResources}
              className="btn-outline inline-flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <ExternalLink className="w-4 h-4" />
              <span>Import Resources</span>
            </motion.button>
          )}
          
          <motion.button
            onClick={() => setIsAdding(true)}
            className="btn-primary inline-flex items-center space-x-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="w-4 h-4" />
            <span>Add Bookmark</span>
          </motion.button>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search bookmarks..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          />
        </div>
        
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="pl-10 pr-8 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value="all">All Categories</option>
            {categories.map(category => (
              <option key={category} value={category}>{category}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Add Bookmark Modal */}
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
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Add Bookmark</h3>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Title *
                  </label>
                  <input
                    type="text"
                    value={newBookmark.title}
                    onChange={(e) => setNewBookmark({ ...newBookmark, title: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Enter bookmark title..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    URL *
                  </label>
                  <input
                    type="url"
                    value={newBookmark.url}
                    onChange={(e) => setNewBookmark({ ...newBookmark, url: e.target.value })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="https://..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Description
                  </label>
                  <textarea
                    value={newBookmark.description}
                    onChange={(e) => setNewBookmark({ ...newBookmark, description: e.target.value })}
                    rows={3}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    placeholder="Brief description..."
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Category
                    </label>
                    <select
                      value={newBookmark.category}
                      onChange={(e) => setNewBookmark({ ...newBookmark, category: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    >
                      {categories.map(category => (
                        <option key={category} value={category}>{category}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Tags (comma separated)
                    </label>
                    <input
                      type="text"
                      value={newBookmark.tags}
                      onChange={(e) => setNewBookmark({ ...newBookmark, tags: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                      placeholder="tag1, tag2, tag3..."
                    />
                  </div>
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
                  onClick={addBookmark}
                  className="btn-primary"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Add Bookmark
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bookmarks Grid */}
      {filteredBookmarks.length === 0 ? (
        <motion.div 
          className="text-center py-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Bookmark className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Bookmarks Found
          </h3>
          <p className="text-gray-600 dark:text-gray-300">
            {searchTerm || selectedCategory !== 'all' 
              ? 'Try adjusting your search or filter criteria.'
              : 'Add bookmarks to save important resources for later.'
            }
          </p>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence>
            {filteredBookmarks.map((bookmark, index) => (
              <motion.div
                key={bookmark.id}
                className="card p-6 hover:shadow-lg transition-shadow"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                whileHover={{ scale: 1.02, y: -5 }}
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white text-lg line-clamp-2 flex-1 mr-2">
                    {bookmark.title}
                  </h3>
                  <div className="flex space-x-2">
                    <motion.a
                      href={bookmark.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-2 text-gray-400 hover:text-blue-500 transition-colors"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <ExternalLink className="w-4 h-4" />
                    </motion.a>
                    <button
                      className="p-2 text-gray-400 hover:text-red-500 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {bookmark.description && (
                  <p className="text-gray-600 dark:text-gray-300 text-sm mb-4 line-clamp-3">
                    {bookmark.description}
                  </p>
                )}

                <div className="flex flex-wrap gap-2 mb-4">
                  {bookmark.tags.map(tag => (
                    <span
                      key={tag}
                      className="px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-full flex items-center space-x-1"
                    >
                      <Tag className="w-3 h-3" />
                      <span>{tag}</span>
                    </span>
                  ))}
                </div>

                <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
                  <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full">
                    {bookmark.category}
                  </span>
                  <span>{bookmark.addedAt.toLocaleDateString()}</span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}