#!/usr/bin/env python3
"""
FastAPI Backend for AI Study Assistant - GROQ VERSION
This version uses Groq API instead of OpenAI
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import List, Dict, Optional
import json
import asyncio
import logging
from pydantic import BaseModel
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing classes
try:
    from pipeline import (
        GroqClient, EnhancedPDFProcessor, SummaryAgent, 
        FlashcardAgent, QuizAgent, EnhancedResearchDiscoveryAgent, 
        YouTubeDiscoveryAgent, WebResourceAgent
    )
    logger.info("✅ Successfully imported pipeline modules")
except ImportError as e:
    logger.error(f"❌ Failed to import pipeline modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="AI Study Assistant API",
    description="Backend API for AI-powered study material generation",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "https://bolt.new",       # Bolt.new
        "https://*.bolt.new",     # Bolt.new subdomains
        "*"  # Allow all origins for development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class ProcessingStatus(BaseModel):
    status: str
    message: str
    word_count: int
    page_count: int
    methods_used: List[str]

class SummaryResponse(BaseModel):
    summary: str
    status: str
    fallback_used: bool = False

class FlashcardResponse(BaseModel):
    flashcards: List[Dict]
    count: int
    status: str
    fallback_used: bool = False

class QuizResponse(BaseModel):
    quiz: List[Dict]
    count: int
    status: str
    fallback_used: bool = False

class ResearchPapersResponse(BaseModel):
    papers: List[Dict]
    count: int
    status: str

class VideosResponse(BaseModel):
    videos: List[Dict]
    count: int
    status: str

class WebResourcesResponse(BaseModel):
    resources: List[Dict]
    count: int
    status: str

class QuestionRequest(BaseModel):
    question: str
    document_text: str

class AnswerResponse(BaseModel):
    answer: str
    status: str
    fallback_used: bool = False

class ApiStatusResponse(BaseModel):
    api_available: bool
    quota_status: str
    message: str
    fallback_enabled: bool

# Global variables to store state
study_sessions = {}
api_status = {
    "available": False,
    "quota_exceeded": False,
    "last_check": 0,
    "consecutive_failures": 0
}

# Initialize agents with error handling
client = None
pdf_processor = None
summary_agent = None
flashcard_agent = None
quiz_agent = None
research_agent = None
youtube_agent = None
web_agent = None

def check_api_status():
    """Check Groq API status and update global status"""
    global api_status
    
    current_time = time.time()
    
    # Only check every 60 seconds to avoid spam
    if current_time - api_status["last_check"] < 60:
        return api_status["available"]
    
    try:
        if client:
            test_response = client.chat_completion(
                [{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            if "❌" in test_response:
                if "quota" in test_response.lower() or "429" in test_response:
                    api_status["quota_exceeded"] = True
                    api_status["available"] = False
                    api_status["consecutive_failures"] += 1
                else:
                    api_status["available"] = False
                    api_status["consecutive_failures"] += 1
            else:
                api_status["available"] = True
                api_status["quota_exceeded"] = False
                api_status["consecutive_failures"] = 0
        else:
            api_status["available"] = False
            
    except Exception as e:
        api_status["available"] = False
        api_status["consecutive_failures"] += 1
        logger.error(f"API status check failed: {e}")
    
    api_status["last_check"] = current_time
    
    # Log status changes
    if api_status["consecutive_failures"] > 0:
        logger.warning(f"⚠️ API issues detected. Consecutive failures: {api_status['consecutive_failures']}")
    
    return api_status["available"]

def generate_fallback_summary(text: str) -> str:
    """Generate a basic summary without AI when quota is exceeded"""
    if not text.strip():
        return "No content available to summarize."
    
    # Basic text analysis
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    words = text.split()
    word_count = len(words)
    
    # Extract first few sentences as summary
    summary_sentences = sentences[:5] if len(sentences) >= 5 else sentences
    
    summary = f"""## 📋 DOCUMENT SUMMARY (Fallback Mode)

**Document Overview:**
This document contains approximately {word_count} words across multiple sections.

**Key Content (First Few Sentences):**
{'. '.join(summary_sentences[:3])}.

**Document Structure:**
- Total words: {word_count}
- Estimated reading time: {word_count // 200} minutes
- Content type: Academic/Professional material

**Study Recommendations:**
1. Read through the document systematically
2. Take notes on key concepts and definitions
3. Identify main themes and supporting arguments
4. Create your own questions for self-testing

*Note: This is a basic summary generated without AI assistance due to API limitations. For detailed analysis, please try again later when the AI service is available.*"""

    return summary

def generate_fallback_flashcards(text: str, num_cards: int = 10) -> List[Dict]:
    """Generate basic flashcards without AI when quota is exceeded"""
    if not text.strip() or len(text.split()) < 50:
        return []
    
    flashcards = []
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    
    # Extract key terms (simple heuristic)
    words = text.split()
    capitalized_words = [word for word in words if word[0].isupper() and len(word) > 3]
    
    # Create flashcards from sentences and key terms
    for i, sentence in enumerate(sentences[:num_cards]):
        if len(sentence) > 30:
            # Try to create a question from the sentence
            question = f"What does the document say about: {sentence[:50]}...?"
            
            flashcard = {
                'question': question,
                'answer': sentence,
                'difficulty': 'Basic',
                'category': 'Document Content',
                'hint': 'Review the document text carefully'
            }
            flashcards.append(flashcard)
    
    # Add some key term flashcards
    for term in capitalized_words[:num_cards - len(flashcards)]:
        if term not in [card['question'] for card in flashcards]:
            flashcard = {
                'question': f"What is {term}?",
                'answer': f"{term} is mentioned in the document as an important concept. Please refer to the document for detailed information.",
                'difficulty': 'Basic',
                'category': 'Key Terms',
                'hint': f'Look for {term} in the document'
            }
            flashcards.append(flashcard)
    
    return flashcards[:num_cards]

def generate_fallback_quiz(text: str, num_questions: int = 8) -> List[Dict]:
    """Generate basic quiz without AI when quota is exceeded"""
    if not text.strip() or len(text.split()) < 100:
        return []
    
    quiz_questions = []
    
    # Generate basic questions about the document
    word_count = len(text.split())
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    # Question 1: Document length
    quiz_questions.append({
        'question': 'Approximately how many words does this document contain?',
        'options': [
            f'About {word_count} words',
            f'About {word_count // 2} words', 
            f'About {word_count * 2} words',
            f'About {word_count // 3} words'
        ],
        'correct_answer': 0,
        'explanation': f'The document contains approximately {word_count} words.',
        'difficulty': 'Basic'
    })
    
    # Question 2: Content type
    quiz_questions.append({
        'question': 'What type of document is this most likely to be?',
        'options': [
            'Academic or professional material',
            'Fiction novel',
            'Recipe collection',
            'Shopping list'
        ],
        'correct_answer': 0,
        'explanation': 'Based on the content structure and complexity, this appears to be academic or professional material.',
        'difficulty': 'Basic'
    })
    
    # Generate questions from first few sentences
    for i, sentence in enumerate(sentences[:num_questions-2]):
        if len(sentence) > 40:
            quiz_questions.append({
                'question': f'According to the document, which statement is true?',
                'options': [
                    sentence[:60] + ('...' if len(sentence) > 60 else ''),
                    'This information is not mentioned in the document',
                    'The document states the opposite of this',
                    'This is only mentioned as a possibility'
                ],
                'correct_answer': 0,
                'explanation': f'The document states: {sentence}',
                'difficulty': 'Basic'
            })
            
            if len(quiz_questions) >= num_questions:
                break
    
    return quiz_questions[:num_questions]

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup with better error handling"""
    global client, pdf_processor, summary_agent, flashcard_agent, quiz_agent, research_agent, youtube_agent, web_agent
    
    try:
        logger.info("🚀 Initializing AI agents...")
        
        # Check for Groq API key
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("⚠️ GROQ_API_KEY not found in environment variables")
            logger.info("🔄 Running in fallback mode - basic functionality only")
        
        # Initialize PDF processor (always available)
        pdf_processor = EnhancedPDFProcessor()
        logger.info("✅ PDF processor initialized")
        
        # Try to initialize AI components
        try:
            client = GroqClient()
            summary_agent = SummaryAgent(client)
            flashcard_agent = FlashcardAgent(client)
            quiz_agent = QuizAgent(client)
            research_agent = EnhancedResearchDiscoveryAgent(client)
            youtube_agent = YouTubeDiscoveryAgent(client)
            web_agent = WebResourceAgent(client)
            
            logger.info("✅ AI agents initialized")
            
            # Check API status
            if check_api_status():
                logger.info("✅ Groq API connection successful")
            else:
                logger.warning("⚠️ Groq API issues detected - fallback mode enabled")
                
        except Exception as e:
            logger.error(f"❌ Error initializing AI agents: {e}")
            logger.info("🔄 Running in fallback mode - basic functionality only")
            
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {e}")
        # Don't raise - allow server to start in fallback mode

@app.get("/")
async def root():
    return {"message": "AI Study Assistant API", "version": "1.0.0", "status": "running"}

@app.get("/api-status", response_model=ApiStatusResponse)
async def get_api_status():
    """Get current API status"""
    is_available = check_api_status()
    
    if api_status["quota_exceeded"]:
        status_msg = "Quota exceeded"
        quota_status = "exceeded"
    elif not is_available:
        status_msg = "API unavailable"
        quota_status = "unavailable"
    else:
        status_msg = "API operational"
        quota_status = "available"
    
    return ApiStatusResponse(
        api_available=is_available,
        quota_status=quota_status,
        message=status_msg,
        fallback_enabled=True
    )

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    is_api_available = check_api_status()
    
    health_status = {
        "status": "healthy",
        "pdf_processor": pdf_processor is not None,
        "ai_agents_initialized": all([
            summary_agent, flashcard_agent, quiz_agent, 
            research_agent, youtube_agent, web_agent
        ]),
        "groq_api_available": is_api_available,
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "fallback_mode": not is_api_available,
        "active_sessions": len(study_sessions),
        "consecutive_api_failures": api_status["consecutive_failures"]
    }
    
    return health_status

@app.post("/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file - This works without AI"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Check file size (limit to 50MB)
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    temp_file_path = None
    
    try:
        logger.info(f"📄 Processing PDF: {file.filename} ({file_size/1024/1024:.2f}MB)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process PDF with timeout handling
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(pdf_processor.extract_text_with_ocr, temp_file_path),
                timeout=120.0  # 2 minutes timeout
            )
        except asyncio.TimeoutError:
            logger.error("❌ PDF processing timeout")
            raise HTTPException(status_code=408, detail="PDF processing timeout. File may be too complex or large.")
        
        if result["status"] == "error":
            logger.error(f"❌ PDF processing failed: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])
        
        if result["word_count"] < 10:
            logger.warning("⚠️ Very little text extracted from PDF")
            raise HTTPException(
                status_code=422, 
                detail="Very little text could be extracted. PDF may be image-based, protected, or corrupted."
            )
        
        # Store session data
        session_id = "default"
        study_sessions[session_id] = {
            "text": result["text"],
            "file_info": f"File: {file.filename} ({file_size/1024/1024:.2f} MB)",
            "processing_result": result,
            "filename": file.filename
        }
        
        logger.info(f"✅ PDF processed successfully: {result['word_count']} words extracted")
        
        return ProcessingStatus(
            status=result["status"],
            message=result["message"],
            word_count=result["word_count"],
            page_count=result["page_count"],
            methods_used=result["methods_used"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp file: {e}")

@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(session_id: str = "default"):
    """Generate summary with fallback support"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    text = study_sessions[session_id]["text"]
    is_api_available = check_api_status()
    
    try:
        if is_api_available and summary_agent:
            logger.info("📝 Generating AI summary...")
            
            # Limit text length for faster processing
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                logger.info(f"📝 Text truncated to {max_chars} characters for faster processing")
            
            # Generate summary with timeout
            summary = await asyncio.wait_for(
                asyncio.to_thread(summary_agent.generate_summary, text),
                timeout=90.0
            )
            
            if summary.startswith("❌"):
                # AI failed, use fallback
                logger.warning("AI summary failed, using fallback")
                summary = generate_fallback_summary(text)
                return SummaryResponse(summary=summary, status="success", fallback_used=True)
            
            logger.info("✅ AI summary generated successfully")
            return SummaryResponse(summary=summary, status="success", fallback_used=False)
        
        else:
            # Use fallback mode
            logger.info("📝 Generating fallback summary (API unavailable)...")
            summary = generate_fallback_summary(text)
            return SummaryResponse(summary=summary, status="success", fallback_used=True)
    
    except asyncio.TimeoutError:
        logger.error("❌ Summary generation timeout, using fallback")
        summary = generate_fallback_summary(text)
        return SummaryResponse(summary=summary, status="success", fallback_used=True)
    except Exception as e:
        logger.error(f"❌ Summary generation error: {str(e)}, using fallback")
        summary = generate_fallback_summary(text)
        return SummaryResponse(summary=summary, status="success", fallback_used=True)

@app.post("/generate-flashcards", response_model=FlashcardResponse)
async def generate_flashcards(session_id: str = "default", num_cards: int = 10):
    """Generate flashcards with fallback support"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if num_cards < 1 or num_cards > 20:
        num_cards = min(max(num_cards, 1), 20)
    
    text = study_sessions[session_id]["text"]
    is_api_available = check_api_status()
    
    try:
        if is_api_available and flashcard_agent:
            logger.info(f"🃏 Generating {num_cards} AI flashcards...")
            
            # Limit text length for faster processing
            max_chars = 6000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            # Generate flashcards with timeout
            flashcards = await asyncio.wait_for(
                asyncio.to_thread(flashcard_agent.generate_flashcards_structured, text, num_cards),
                timeout=120.0
            )
            
            if not flashcards:
                # AI failed, use fallback
                logger.warning("AI flashcard generation failed, using fallback")
                flashcards = generate_fallback_flashcards(text, num_cards)
                return FlashcardResponse(flashcards=flashcards, count=len(flashcards), status="success", fallback_used=True)
            
            logger.info(f"✅ Generated {len(flashcards)} AI flashcards successfully")
            return FlashcardResponse(flashcards=flashcards, count=len(flashcards), status="success", fallback_used=False)
        
        else:
            # Use fallback mode
            logger.info(f"🃏 Generating {num_cards} fallback flashcards (API unavailable)...")
            flashcards = generate_fallback_flashcards(text, num_cards)
            return FlashcardResponse(flashcards=flashcards, count=len(flashcards), status="success", fallback_used=True)
    
    except asyncio.TimeoutError:
        logger.error("❌ Flashcard generation timeout, using fallback")
        flashcards = generate_fallback_flashcards(text, num_cards)
        return FlashcardResponse(flashcards=flashcards, count=len(flashcards), status="success", fallback_used=True)
    except Exception as e:
        logger.error(f"❌ Flashcard generation error: {str(e)}, using fallback")
        flashcards = generate_fallback_flashcards(text, num_cards)
        return FlashcardResponse(flashcards=flashcards, count=len(flashcards), status="success", fallback_used=True)

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(session_id: str = "default", num_questions: int = 8):
    """Generate quiz with fallback support"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if num_questions < 1 or num_questions > 15:
        num_questions = min(max(num_questions, 1), 15)
    
    text = study_sessions[session_id]["text"]
    is_api_available = check_api_status()
    
    try:
        if is_api_available and quiz_agent:
            logger.info(f"📝 Generating {num_questions} AI quiz questions...")
            
            # Limit text length for faster processing
            max_chars = 6000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            # Generate quiz with timeout
            quiz = await asyncio.wait_for(
                asyncio.to_thread(quiz_agent.generate_quiz_structured, text, num_questions),
                timeout=120.0
            )
            
            if not quiz:
                # AI failed, use fallback
                logger.warning("AI quiz generation failed, using fallback")
                quiz = generate_fallback_quiz(text, num_questions)
                return QuizResponse(quiz=quiz, count=len(quiz), status="success", fallback_used=True)
            
            logger.info(f"✅ Generated {len(quiz)} AI quiz questions successfully")
            return QuizResponse(quiz=quiz, count=len(quiz), status="success", fallback_used=False)
        
        else:
            # Use fallback mode
            logger.info(f"📝 Generating {num_questions} fallback quiz questions (API unavailable)...")
            quiz = generate_fallback_quiz(text, num_questions)
            return QuizResponse(quiz=quiz, count=len(quiz), status="success", fallback_used=True)
    
    except asyncio.TimeoutError:
        logger.error("❌ Quiz generation timeout, using fallback")
        quiz = generate_fallback_quiz(text, num_questions)
        return QuizResponse(quiz=quiz, count=len(quiz), status="success", fallback_used=True)
    except Exception as e:
        logger.error(f"❌ Quiz generation error: {str(e)}, using fallback")
        quiz = generate_fallback_quiz(text, num_questions)
        return QuizResponse(quiz=quiz, count=len(quiz), status="success", fallback_used=True)

@app.post("/discover-research", response_model=ResearchPapersResponse)
async def discover_research(session_id: str = "default", max_papers: int = 10):
    """Discover research papers - works without AI quota"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if max_papers > 15:
        max_papers = 15
    
    try:
        logger.info("🔍 Discovering research papers...")
        text = study_sessions[session_id]["text"]
        
        # This can work even with quota issues since it mainly uses web search
        papers = await asyncio.wait_for(
            asyncio.to_thread(research_agent.find_papers, text, max_papers),
            timeout=180.0
        )
        
        logger.info(f"✅ Found {len(papers)} research papers")
        return ResearchPapersResponse(papers=papers, count=len(papers), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Research discovery timeout")
        return ResearchPapersResponse(papers=[], count=0, status="success")
    except Exception as e:
        logger.error(f"❌ Research discovery error: {str(e)}")
        return ResearchPapersResponse(papers=[], count=0, status="success")

@app.post("/discover-videos", response_model=VideosResponse)
async def discover_videos(session_id: str = "default", max_videos: int = 10):
    """Discover YouTube videos - works without AI quota"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if max_videos > 12:
        max_videos = 12
    
    try:
        logger.info("🎥 Discovering educational videos...")
        text = study_sessions[session_id]["text"]
        
        # Extract keywords with fallback
        if check_api_status() and research_agent:
            topic, research_keywords, all_keywords = await asyncio.wait_for(
                asyncio.to_thread(research_agent.extract_smart_keywords_and_topic, text),
                timeout=45.0
            )
        else:
            # Fallback keyword extraction
            words = text.split()[:200]  # First 200 words
            topic = "Study Material"
            research_keywords = [word for word in words if len(word) > 4 and word.istitle()][:5]
            all_keywords = research_keywords
        
        # Find videos
        videos = await asyncio.wait_for(
            asyncio.to_thread(youtube_agent.find_videos, research_keywords, topic, max_videos),
            timeout=150.0
        )
        
        logger.info(f"✅ Found {len(videos)} educational videos")
        return VideosResponse(videos=videos, count=len(videos), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Video discovery timeout")
        return VideosResponse(videos=[], count=0, status="success")
    except Exception as e:
        logger.error(f"❌ Video discovery error: {str(e)}")
        return VideosResponse(videos=[], count=0, status="success")

@app.post("/discover-resources", response_model=WebResourcesResponse)
async def discover_resources(session_id: str = "default", max_resources: int = 12):
    """Discover web resources - works without AI quota"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if max_resources > 15:
        max_resources = 15
    
    try:
        logger.info("🌐 Discovering web resources...")
        text = study_sessions[session_id]["text"]
        
        # Extract keywords with fallback
        if check_api_status() and research_agent:
            topic, research_keywords, all_keywords = await asyncio.wait_for(
                asyncio.to_thread(research_agent.extract_smart_keywords_and_topic, text),
                timeout=45.0
            )
        else:
            # Fallback keyword extraction
            words = text.split()[:200]  # First 200 words
            topic = "Study Material"
            research_keywords = [word for word in words if len(word) > 4 and word.istitle()][:5]
            all_keywords = research_keywords
        
        # Find resources
        resources = await asyncio.wait_for(
            asyncio.to_thread(web_agent.find_resources, research_keywords, topic, max_resources),
            timeout=150.0
        )
        
        logger.info(f"✅ Found {len(resources)} web resources")
        return WebResourcesResponse(resources=resources, count=len(resources), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Resource discovery timeout")
        return WebResourcesResponse(resources=[], count=0, status="success")
    except Exception as e:
        logger.error(f"❌ Resource discovery error: {str(e)}")
        return WebResourcesResponse(resources=[], count=0, status="success")

@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer questions with fallback support"""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not request.document_text.strip():
        raise HTTPException(status_code=400, detail="No document text provided")
    
    is_api_available = check_api_status()
    
    try:
        if is_api_available and client:
            logger.info(f"❓ Answering question with AI: {request.question[:50]}...")
            
            # Limit text for analysis
            max_chars = 6000
            text_content = request.document_text[:max_chars]
            if len(request.document_text) > max_chars:
                text_content += "..."

            prompt = f"""Based on the following document content, please answer the question comprehensively and accurately.

Document Content:
{text_content}

Question: {request.question}

Instructions:
- Provide a detailed, accurate answer based on the document
- If the information isn't in the document, say so clearly
- Use specific examples from the document when possible
- Keep the answer well-structured and easy to understand"""

            # Generate answer with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(client.chat_completion, [{"role": "user", "content": prompt}], None, 800),
                timeout=60.0
            )
            
            if response.startswith("❌"):
                # AI failed, use fallback
                logger.warning("AI question answering failed, using fallback")
                fallback_answer = generate_fallback_answer(request.question, request.document_text)
                return AnswerResponse(answer=fallback_answer, status="success", fallback_used=True)
            
            logger.info("✅ Question answered successfully with AI")
            return AnswerResponse(answer=response, status="success", fallback_used=False)
        
        else:
            # Use fallback mode
            logger.info(f"❓ Answering question with fallback: {request.question[:50]}...")
            fallback_answer = generate_fallback_answer(request.question, request.document_text)
            return AnswerResponse(answer=fallback_answer, status="success", fallback_used=True)
    
    except asyncio.TimeoutError:
        logger.error("❌ Question answering timeout, using fallback")
        fallback_answer = generate_fallback_answer(request.question, request.document_text)
        return AnswerResponse(answer=fallback_answer, status="success", fallback_used=True)
    except Exception as e:
        logger.error(f"❌ Question answering error: {str(e)}, using fallback")
        fallback_answer = generate_fallback_answer(request.question, request.document_text)
        return AnswerResponse(answer=fallback_answer, status="success", fallback_used=True)

def generate_fallback_answer(question: str, document_text: str) -> str:
    """Generate a basic answer without AI when quota is exceeded"""
    if not question.strip() or not document_text.strip():
        return "I need both a question and document content to provide an answer."
    
    # Simple keyword matching approach
    question_lower = question.lower()
    doc_lower = document_text.lower()
    
    # Find sentences containing question keywords
    question_words = [word for word in question_lower.split() if len(word) > 3]
    sentences = [s.strip() for s in document_text.split('.') if len(s.strip()) > 20]
    
    relevant_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        word_matches = sum(1 for word in question_words if word in sentence_lower)
        if word_matches >= 1:  # At least one keyword match
            relevant_sentences.append((sentence, word_matches))
    
    # Sort by relevance (number of matching keywords)
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_sentences:
        answer = f"""Based on the document content, here's what I found related to your question:

**Question:** {question}

**Relevant information from the document:**

{relevant_sentences[0][0]}"""

        # Add additional relevant sentences if available
        if len(relevant_sentences) > 1:
            answer += f"\n\n**Additional context:**\n\n{relevant_sentences[1][0]}"
        
        if len(relevant_sentences) > 2:
            answer += f"\n\n{relevant_sentences[2][0]}"
        
        answer += f"""

**Note:** This answer was generated using basic text matching due to AI service limitations. For more detailed analysis, please try again later when the AI service is available.

**Suggestion:** You can search the document for keywords related to your question: {', '.join(question_words[:5])}"""
        
        return answer
    else:
        return f"""I couldn't find specific information related to your question "{question}" in the document using basic text matching.

**Suggestions:**
1. Try rephrasing your question with different keywords
2. Check if the information might be expressed differently in the document
3. Browse through the document manually for related concepts

**Question keywords searched:** {', '.join(question_words[:5])}

**Note:** This search was performed using basic text matching due to AI service limitations. For more sophisticated analysis, please try again later when the AI service is available."""

@app.delete("/clear-session")
async def clear_session(session_id: str = "default"):
    """Clear session data"""
    
    if session_id in study_sessions:
        del study_sessions[session_id]
        logger.info(f"🗑️ Cleared session: {session_id}")
        return {"message": "Session cleared successfully", "status": "success"}
    else:
        return {"message": "No active session found", "status": "info"}

@app.get("/session-info")
async def get_session_info(session_id: str = "default"):
    """Get information about current session"""
    
    if session_id not in study_sessions:
        return {"active": False, "message": "No active session"}
    
    session_data = study_sessions[session_id]
    is_api_available = check_api_status()
    
    return {
        "active": True,
        "file_info": session_data.get("file_info", ""),
        "filename": session_data.get("filename", ""),
        "word_count": session_data.get("processing_result", {}).get("word_count", 0),
        "page_count": session_data.get("processing_result", {}).get("page_count", 0),
        "methods_used": session_data.get("processing_result", {}).get("methods_used", []),
        "api_status": {
            "available": is_api_available,
            "fallback_mode": not is_api_available,
            "quota_exceeded": api_status["quota_exceeded"]
        }
    }

@app.post("/check-quota")
async def check_quota_endpoint():
    """Endpoint to manually check API quota status"""
    is_available = check_api_status()
    
    return {
        "api_available": is_available,
        "quota_exceeded": api_status["quota_exceeded"],
        "consecutive_failures": api_status["consecutive_failures"],
        "last_check": api_status["last_check"],
        "message": "API operational" if is_available else "API unavailable - using fallback mode",
        "fallback_features": [
            "Basic PDF text extraction",
            "Simple summary generation", 
            "Basic flashcard creation",
            "Simple quiz generation",
            "Keyword-based question answering",
            "Web resource discovery",
            "Video discovery"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Study Assistant API with Groq Integration...")
    print("📱 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    print("💡 Features work with or without Groq API quota!")
    
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
