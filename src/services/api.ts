// src/services/api.ts
import axios from 'axios';
import type {
  ProcessingStatus,
  StudySession,
  Flashcard,
  QuizQuestion,
  ResearchPaper,
  YouTubeVideo,
  WebResource
} from '../types';

// Prefer an explicit env var (Vite) when available, otherwise use '/api' so you can set up a Vite proxy.
// Example .env: VITE_API_BASE=https://<your-forwarded>-8000.app.github.dev
const API_BASE = (import.meta?.env?.VITE_API_BASE as string) || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60 seconds for file processing
});

// Request interceptor for logging
api.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    throw error;
  }
);

export const apiService = {
  // Health check
  async healthCheck(): Promise<{ status: string }> {
    const response = await api.get('/health');
    return response.data;
  },

  // Upload PDF
  async uploadPDF(file: File): Promise<ProcessingStatus> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload-pdf', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Generate summary
  async generateSummary(sessionId: string = 'default'): Promise<{ summary: string; status: string }> {
    const response = await api.post(`/generate-summary?session_id=${sessionId}`);
    return response.data;
  },

  // Generate flashcards
  async generateFlashcards(sessionId: string = 'default', numCards: number = 10): Promise<{ flashcards: Flashcard[]; count: number; status: string }> {
    const response = await api.post(`/generate-flashcards?session_id=${sessionId}&num_cards=${numCards}`);
    return response.data;
  },

  // Generate quiz
  async generateQuiz(sessionId: string = 'default', numQuestions: number = 8): Promise<{ quiz: QuizQuestion[]; count: number; status: string }> {
    const response = await api.post(`/generate-quiz?session_id=${sessionId}&num_questions=${numQuestions}`);
    return response.data;
  },

  // Discover research papers
  async discoverResearch(sessionId: string = 'default', maxPapers: number = 10): Promise<{ papers: ResearchPaper[]; count: number; status: string }> {
    const response = await api.post(`/discover-research?session_id=${sessionId}&max_papers=${maxPapers}`);
    return response.data;
  },

  // Discover YouTube videos
  async discoverVideos(sessionId: string = 'default', maxVideos: number = 10): Promise<{ videos: YouTubeVideo[]; count: number; status: string }> {
    const response = await api.post(`/discover-videos?session_id=${sessionId}&max_videos=${maxVideos}`);
    return response.data;
  },

  // Discover web resources
  async discoverResources(sessionId: string = 'default', maxResources: number = 12): Promise<{ resources: WebResource[]; count: number; status: string }> {
    const response = await api.post(`/discover-resources?session_id=${sessionId}&max_resources=${maxResources}`);
    return response.data;
  },

  // Ask question
  async askQuestion(question: string, documentText: string): Promise<{ answer: string; status: string }> {
    const response = await api.post('/ask-question', {
      question,
      document_text: documentText,
    });
    return response.data;
  },

  // Clear session
  async clearSession(sessionId: string = 'default'): Promise<{ message: string; status: string }> {
    const response = await api.delete(`/clear-session?session_id=${sessionId}`);
    return response.data;
  },

  // Get session info
  async getSessionInfo(sessionId: string = 'default'): Promise<StudySession> {
    const response = await api.get(`/session-info?session_id=${sessionId}`);
    return response.data;
  },

  // Generate presentation
  async generatePresentation(data: {
    topic: string;
    audience: string;
    duration: number;
    theme: string;
  }): Promise<any> {
    const response = await api.post('/generate-presentation', data);
    return response.data;
  },
};
