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
  studyNotes: StudyNote[];
  citations: Citation[];
  studyPlan: StudyPlan | null;
  conceptMap: ConceptMap | null;
  studySessions: StudySession[];
  studyGoals: StudyGoal[];
  bookmarks: BookmarkItem[];
  currentStudySession: StudySession | null;
  timerState: {
    isRunning: boolean;
    timeLeft: number;
    mode: 'pomodoro' | 'break' | 'custom';
    cycles: number;
  };
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
  | { type: 'CLEAR_SESSION' }
  | { type: 'ADD_STUDY_NOTE'; payload: StudyNote }
  | { type: 'UPDATE_STUDY_NOTE'; payload: StudyNote }
  | { type: 'DELETE_STUDY_NOTE'; payload: string }
  | { type: 'ADD_CITATION'; payload: Citation }
  | { type: 'SET_STUDY_PLAN'; payload: StudyPlan }
  | { type: 'UPDATE_STUDY_PLAN'; payload: Partial<StudyPlan> }
  | { type: 'SET_CONCEPT_MAP'; payload: ConceptMap }
  | { type: 'ADD_STUDY_SESSION'; payload: StudySession }
  | { type: 'SET_CURRENT_STUDY_SESSION'; payload: StudySession | null }
  | { type: 'ADD_STUDY_GOAL'; payload: StudyGoal }
  | { type: 'UPDATE_STUDY_GOAL'; payload: StudyGoal }
  | { type: 'ADD_BOOKMARK'; payload: BookmarkItem }
  | { type: 'UPDATE_TIMER_STATE'; payload: Partial<typeof initialState.timerState> };

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
  studyNotes: [],
  citations: [],
  studyPlan: null,
  conceptMap: null,
  studySessions: [],
  studyGoals: [],
  bookmarks: [],
  currentStudySession: null,
  timerState: {
    isRunning: false,
    timeLeft: 25 * 60, // 25 minutes in seconds
    mode: 'pomodoro',
    cycles: 0,
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
    case 'ADD_STUDY_NOTE':
      return { ...state, studyNotes: [...state.studyNotes, action.payload] };
    case 'UPDATE_STUDY_NOTE':
      return { 
        ...state, 
        studyNotes: state.studyNotes.map(note => 
          note.id === action.payload.id ? action.payload : note
        )
      };
    case 'DELETE_STUDY_NOTE':
      return { 
        ...state, 
        studyNotes: state.studyNotes.filter(note => note.id !== action.payload)
      };
    case 'ADD_CITATION':
      return { ...state, citations: [...state.citations, action.payload] };
    case 'SET_STUDY_PLAN':
      return { ...state, studyPlan: action.payload };
    case 'UPDATE_STUDY_PLAN':
      return { 
        ...state, 
        studyPlan: state.studyPlan ? { ...state.studyPlan, ...action.payload } : null
      };
    case 'SET_CONCEPT_MAP':
      return { ...state, conceptMap: action.payload };
    case 'ADD_STUDY_SESSION':
      return { ...state, studySessions: [...state.studySessions, action.payload] };
    case 'SET_CURRENT_STUDY_SESSION':
      return { ...state, currentStudySession: action.payload };
    case 'ADD_STUDY_GOAL':
      return { ...state, studyGoals: [...state.studyGoals, action.payload] };
    case 'UPDATE_STUDY_GOAL':
      return { 
        ...state, 
        studyGoals: state.studyGoals.map(goal => 
          goal.id === action.payload.id ? action.payload : goal
        )
      };
    case 'ADD_BOOKMARK':
      return { ...state, bookmarks: [...state.bookmarks, action.payload] };
    case 'UPDATE_TIMER_STATE':
      return { 
        ...state, 
        timerState: { ...state.timerState, ...action.payload }
      };
    case 'CLEAR_SESSION':
      return {
        ...initialState,
        theme: state.theme,
        chatMessages: [],
        studyNotes: state.studyNotes,
        citations: state.citations,
        studyGoals: state.studyGoals,
        bookmarks: state.bookmarks,
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