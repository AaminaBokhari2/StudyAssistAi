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

export type TabType = 'summary' | 'flashcards' | 'quiz' | 'qa' | 'research' | 'videos' | 'resources' | 'notes' | 'citations' | 'planner' | 'concepts' | 'analytics' | 'bookmarks' | 'timer';

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

export interface StudyPlan {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  topics: StudyTopic[];
  progress: number;
  createdAt: Date;
}

export interface StudyTopic {
  id: string;
  title: string;
  description: string;
  estimatedTime: string;
  completed: boolean;
  resources: string[];
  notes: string;
}

export interface Citation {
  id: string;
  type: 'APA' | 'MLA' | 'Chicago' | 'Harvard';
  text: string;
  source: string;
  authors: string[];
  year: string;
  title: string;
}

export interface StudyNote {
  id: string;
  title: string;
  content: string;
  tags: string[];
  color: string;
  createdAt: Date;
  updatedAt: Date;
  category: 'Important' | 'Question' | 'Summary' | 'Idea' | 'Todo';
}

export interface ConceptMap {
  id: string;
  title: string;
  nodes: ConceptNode[];
  connections: ConceptConnection[];
}

export interface ConceptNode {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  size: number;
}

export interface ConceptConnection {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface StudySession {
  id: string;
  startTime: Date;
  endTime?: Date;
  duration: number;
  topic: string;
  focusScore: number;
  completedTasks: number;
  notes: string;
}

export interface StudyGoal {
  id: string;
  title: string;
  description: string;
  targetDate: Date;
  progress: number;
  completed: boolean;
  priority: 'Low' | 'Medium' | 'High';
  category: string;
}

export interface BookmarkItem {
  id: string;
  title: string;
  url: string;
  description: string;
  tags: string[];
  category: string;
  addedAt: Date;
  thumbnail?: string;
}