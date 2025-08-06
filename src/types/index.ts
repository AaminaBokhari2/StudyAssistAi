export interface ProcessingStatus {
  status: string;
  message: string;
  word_count: number;
  page_count: number;
  methods_used: string[];
}

export interface StudySession {
  active: boolean;
  file_info?: string;
  word_count?: number;
  page_count?: number;
  methods_used?: string[];
}

export interface Flashcard {
  question: string;
  answer: string;
  category?: string;
  difficulty?: 'Easy' | 'Medium' | 'Hard';
  hint?: string;
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correct_answer: number;
  explanation?: string;
  difficulty?: 'Easy' | 'Medium' | 'Hard';
}

export interface ResearchPaper {
  title: string;
  authors: string;
  year: string;
  source: string;
  abstract?: string;
  url?: string;
  relevance_score?: number;
  relevance_label?: string;
  citation_count?: number;
  fields_of_study?: string[];
  categories?: string[];
}

export interface YouTubeVideo {
  title: string;
  channel: string;
  duration: string;
  views: string;
  description?: string;
  url: string;
  educational_score?: string;
}

export interface WebResource {
  title: string;
  type: string;
  source: string;
  description?: string;
  url: string;
  quality_score?: string;
}

export interface ApiResponse<T> {
  status: string;
  data?: T;
  message?: string;
}

export type TabType = 'summary' | 'flashcards' | 'quiz' | 'qa' | 'research' | 'videos' | 'resources';

export interface QuizState {
  currentQuestion: number;
  answers: Record<number, string>;
  score: number;
  completed: boolean;
  showExplanation: boolean;
}

export interface FlashcardState {
  currentCard: number;
  isFlipped: boolean;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}