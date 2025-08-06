import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import type { 
  StudySession, 
  Flashcard, 
  QuizQuestion, 
  ResearchPaper, 
  YouTubeVideo, 
  WebResource, 
  TabType, 
  QuizState, 
  FlashcardState,
  ChatMessage
} from '../types';

interface AppState {
  theme: 'light' | 'dark';
  session: StudySession;
  currentTab: TabType;
  summary: string;
  flashcards: Flashcard[];
  quiz: QuizQuestion[];
  researchPapers: ResearchPaper[];
  youtubeVideos: YouTubeVideo[];
  webResources: WebResource[];
  chatMessages: ChatMessage[];
  documentText: string;
  isLoading: boolean;
  error: string | null;
  quizState: QuizState;
  flashcardState: FlashcardState;
}

type AppAction =
  | { type: 'TOGGLE_THEME' }
  | { type: 'SET_SESSION'; payload: StudySession }
  | { type: 'SET_TAB'; payload: TabType }
  | { type: 'SET_SUMMARY'; payload: string }
  | { type: 'SET_FLASHCARDS'; payload: Flashcard[] }
  | { type: 'SET_QUIZ'; payload: QuizQuestion[] }
  | { type: 'SET_RESEARCH_PAPERS'; payload: ResearchPaper[] }
  | { type: 'SET_YOUTUBE_VIDEOS'; payload: YouTubeVideo[] }
  | { type: 'SET_WEB_RESOURCES'; payload: WebResource[] }
  | { type: 'ADD_CHAT_MESSAGE'; payload: ChatMessage }
  | { type: 'SET_DOCUMENT_TEXT'; payload: string }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'UPDATE_QUIZ_STATE'; payload: Partial<QuizState> }
  | { type: 'UPDATE_FLASHCARD_STATE'; payload: Partial<FlashcardState> }
  | { type: 'CLEAR_SESSION' };

const initialState: AppState = {
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'light',
  session: { active: false },
  currentTab: 'summary',
  summary: '',
  flashcards: [],
  quiz: [],
  researchPapers: [],
  youtubeVideos: [],
  webResources: [],
  chatMessages: [],
  documentText: '',
  isLoading: false,
  error: null,
  quizState: {
    currentQuestion: 0,
    answers: {},
    score: 0,
    completed: false,
    showExplanation: false,
  },
  flashcardState: {
    currentCard: 0,
    isFlipped: false,
  },
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'TOGGLE_THEME':
      const newTheme = state.theme === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme', newTheme);
      return { ...state, theme: newTheme };
    case 'SET_SESSION':
      return { ...state, session: action.payload };
    case 'SET_TAB':
      return { ...state, currentTab: action.payload };
    case 'SET_SUMMARY':
      return { ...state, summary: action.payload };
    case 'SET_FLASHCARDS':
      return { 
        ...state, 
        flashcards: action.payload,
        flashcardState: { currentCard: 0, isFlipped: false }
      };
    case 'SET_QUIZ':
      return { 
        ...state, 
        quiz: action.payload,
        quizState: {
          currentQuestion: 0,
          answers: {},
          score: 0,
          completed: false,
          showExplanation: false,
        }
      };
    case 'SET_RESEARCH_PAPERS':
      return { ...state, researchPapers: action.payload };
    case 'SET_YOUTUBE_VIDEOS':
      return { ...state, youtubeVideos: action.payload };
    case 'SET_WEB_RESOURCES':
      return { ...state, webResources: action.payload };
    case 'ADD_CHAT_MESSAGE':
      return { ...state, chatMessages: [...state.chatMessages, action.payload] };
    case 'SET_DOCUMENT_TEXT':
      return { ...state, documentText: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'UPDATE_QUIZ_STATE':
      return { 
        ...state, 
        quizState: { ...state.quizState, ...action.payload }
      };
    case 'UPDATE_FLASHCARD_STATE':
      return { 
        ...state, 
        flashcardState: { ...state.flashcardState, ...action.payload }
      };
    case 'CLEAR_SESSION':
      return {
        ...initialState,
        theme: state.theme,
        chatMessages: [],
      };
    default:
      return state;
  }
}

const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
} | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}