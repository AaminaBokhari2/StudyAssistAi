#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
import time
import json
import re
import requests
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from collections import Counter
from urllib.parse import urlparse
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Constants for content truncation
MAX_SUMMARY_CHARS = 8000
MAX_FLASHCARD_CHARS = 7000
MAX_QUIZ_CHARS = 7000
MAX_RESEARCH_KEYWORDS_CHARS = 6000
MAX_PRESENTATION_CONTENT_CHARS = 15000 # Max characters to pass to LLM for presentation generation

##### GROQ CLIENT WITH IMPROVED ERROR HANDLING #####
class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # This will allow the app to run in fallback mode for AI features
            print("⚠️ GROQ_API_KEY not found. AI features will run in fallback mode.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
        self.model_fallbacks = [
            "llama3-70b-8192",      # Larger model option, prioritize this
            "llama3-8b-8192"        # New fallback option
        ]
    def chat_completion(self, messages: List[Dict], model: str = None, max_tokens: int = None, retry_count: int = 3) -> str:
        if not self.client:
            return "❌ Groq API client not initialized. Running in fallback mode."
        if model is None:
            models_to_try = self.model_fallbacks
        else:
            models_to_try = [model] + [m for m in self.model_fallbacks if m != model]
        for attempt in range(retry_count):
            for model_name in models_to_try:
                try:
                    print(f"🤖 Using model: {model_name} (attempt {attempt + 1})")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"⏱️ Error with {model_name}: {str(e)}")
                    if attempt < retry_count - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    continue
        return "❌ All AI models failed. Please check your Groq API key and internet connection."

##### ENHANCED PDF PROCESSOR WITH BETTER ERROR HANDLING #####
class EnhancedPDFProcessor:
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        print(f"🔍 Tesseract available: {self.tesseract_available}")
    def _check_tesseract(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            print(f"⚠️ Tesseract not available: {e}")
            return False
    def extract_text_with_ocr(self, file_path: str, max_pages: int = 20) -> Dict[str, any]:
        result = {
            "text": "",
            "page_count": 0,
            "extracted_pages": 0,
            "ocr_pages": 0,
            "word_count": 0,
            "status": "success",
            "methods_used": [],
            "message": "",
            "error_details": []
        }
        if not os.path.exists(file_path):
            result["status"] = "error"
            result["message"] = f"File not found: {file_path}"
            return result
        try:
            print(f"📄 Processing PDF: {file_path}")
            text_content = ""
            ocr_content = ""
            # Try pdfplumber first
            try:
                with pdfplumber.open(file_path) as pdf:
                    result["page_count"] = len(pdf.pages)
                    pages_to_process = min(result["page_count"], max_pages)
                    print(f"📊 PDF has {result['page_count']} pages, processing {pages_to_process}")
                    for page_num, page in enumerate(pdf.pages[:pages_to_process], 1):
                        try:
                            page_text = page.extract_text()
                            if page_text and len(page_text.strip()) > 20:
                                text_content += f"\n--- Page {page_num} ---\n{page_text.strip()}\n"
                                result["extracted_pages"] += 1
                                print(f"✅ Extracted text from page {page_num}: {len(page_text)} chars")
                            else:
                                print(f"⚠️ Page {page_num}: No meaningful text found")
                        except Exception as e:
                            result["error_details"].append(f"Page {page_num}: {str(e)}")
                            continue
                    if result["extracted_pages"] > 0:
                        result["methods_used"].append("text_extraction")
                        print(f"✅ Successfully extracted text from {result['extracted_pages']} pages")
            except Exception as e:
                result["error_details"].append(f"PDFPlumber error: {str(e)}")
                print(f"❌ PDFPlumber failed: {e}")
            # Try OCR if text extraction failed or yielded poor results
            if (result["extracted_pages"] == 0 or
                result["extracted_pages"] < result["page_count"] * 0.3):
                if self.tesseract_available:
                    print("🔍 Attempting OCR extraction...")
                    try:
                        ocr_content = self._extract_with_ocr(file_path, max_pages=min(5, result["page_count"]))
                        if ocr_content:
                            result["methods_used"].append("ocr")
                            result["ocr_pages"] = ocr_content.count("--- Page")
                            print(f"✅ OCR extracted text from {result['ocr_pages']} pages")
                    except Exception as e:
                        result["error_details"].append(f"OCR error: {str(e)}")
                        print(f"❌ OCR failed: {e}")
                else:
                    result["error_details"].append("OCR not available (Tesseract not installed)")
            # Combine results
            final_text = text_content + ocr_content
            result["text"] = final_text.strip()
            result["word_count"] = len(final_text.split())
            # Set final status and message
            if result["word_count"] > 50:
                methods = " + ".join(result["methods_used"])
                result["message"] = f"✅ Successfully extracted {result['word_count']} words using: {methods}"
                result["status"] = "success"
            elif result["word_count"] > 0:
                result["status"] = "warning"
                result["message"] = f"⚠️ Limited content extracted ({result['word_count']} words). PDF may be image-based or have formatting issues."
            else:
                result["status"] = "error"
                result["message"] = "❌ No text could be extracted. PDF might be image-based, protected, or corrupted."
            print(f"📊 Final result: {result['status']} - {result['word_count']} words")
            return result
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"❌ Critical error processing PDF: {str(e)}"
            result["error_details"].append(str(e))
            return result
    def _extract_with_ocr(self, file_path: str, max_pages: int = 5) -> str:
        if not self.tesseract_available:
            return ""
        ocr_text = ""
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"🖼️ Converting PDF to images for OCR (max {max_pages} pages)...")
            images = convert_from_path(
                file_path,
                first_page=1,
                last_page=max_pages,
                output_folder=temp_dir,
                dpi=200
            )
            for i, image in enumerate(images, 1):
                try:
                    print(f"🔍 Processing image {i} with OCR...")
                    page_text = pytesseract.image_to_string(
                        image,
                        lang='eng',
                        config='--psm 6'
                    )
                    if page_text.strip():
                        ocr_text += f"\n--- Page {i} (OCR) ---\n{page_text.strip()}\n"
                        print(f"✅ OCR page {i}: {len(page_text)} chars")
                except Exception as e:
                    print(f"❌ OCR failed on page {i}: {e}")
                    continue
            return ocr_text
        except Exception as e:
            print(f"❌ OCR process failed: {e}")
            return ""
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

##### ENHANCED STUDY AGENTS WITH FIXED PROMPTS #####
class SummaryAgent:
    def __init__(self, client: GroqClient):
        self.client = client
    def generate_summary(self, text: str) -> str:
        if not text.strip():
            return "❌ No content available to summarize."
        if len(text.split()) < 10:
            return "⚠️ Content too short for meaningful summary."
        # Truncate text if too long
        max_chars = MAX_SUMMARY_CHARS
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        prompt = f"""Create a comprehensive, well-structured summary of the following academic content.
Document Content:
{text}
Please structure your summary as follows:
## 📋 DOCUMENT OVERVIEW
- Main topic and purpose
- Document type and scope
## 🎯 KEY CONCEPTS & DEFINITIONS
List the 5-8 most important concepts with brief explanations
## 📝 DETAILED SUMMARY
Write 2-3 paragraphs providing a comprehensive overview including:
- Main arguments or points
- Supporting evidence or examples
- Relationships between concepts
- Conclusions or implications
## 🔑 CRITICAL TAKEAWAYS
List 4-6 essential points that students must remember
## 📚 STUDY FOCUS AREAS
Highlight areas that deserve extra attention
Make the summary engaging, clear, and educational."""
        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=1500)
            if response.startswith("❌"):
                return f"❌ Summary generation failed: {response}"
            return response
        except Exception as e:
            return f"❌ Summary generation failed: {str(e)}"

class FlashcardAgent:
    def __init__(self, client: GroqClient):
        self.client = client
    def generate_flashcards_structured(self, text: str, num_cards=10) -> List[Dict]:
        """Generate structured flashcards data for the app interface"""
        if not text.strip():
            return []
        if len(text.split()) < 20:
            return []
        # Truncate text if too long
        max_chars = MAX_FLASHCARD_CHARS
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        prompt = f"""Create {num_cards} high-quality study flashcards based on the following content. Return ONLY a valid JSON array with no additional text.
Content:
{text}
Return format (MUST be valid JSON):
[
  {{
      "question": "Clear, specific question",
      "answer": "Comprehensive answer with examples",
      "difficulty": "Basic",
      "category": "Main topic category",
      "hint": "Optional memory aid or hint"
  }},
  {{
      "question": "Another clear question",
      "answer": "Another comprehensive answer",
      "difficulty": "Intermediate",
       "category": "Topic category",
      "hint": "Memory aid"
  }}
]
Guidelines:
- Create diverse question types (definitions, applications, comparisons)
- Test understanding, not just memorization
- Include relevant examples in answers
- Mix difficulty levels: Basic, Intermediate, Advanced
- Keep questions clear and answers comprehensive"""
        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=2500)
            # Clean response to extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            # Try to parse JSON response
            try:
                flashcards_data = json.loads(response)
                if isinstance(flashcards_data, list) and len(flashcards_data) > 0:
                    # Validate structure
                    valid_cards = []
                    for card in flashcards_data:
                        if isinstance(card, dict) and 'question' in card and 'answer' in card:
                            # Ensure all required fields
                            valid_card = {
                                'question': str(card.get('question', '')),
                                'answer': str(card.get('answer', '')),
                                'difficulty': card.get('difficulty', 'Basic'),
                                'category': card.get('category', 'General'),
                                'hint': card.get('hint', '')
                            }
                            valid_cards.append(valid_card)
                    if valid_cards:
                        print(f"✅ Generated {len(valid_cards)} flashcards")
                        return valid_cards
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"Response: {response[:200]}...")
            # Fallback: generate basic flashcards
            return self._generate_basic_flashcards(text, num_cards)
        except Exception as e:
            print(f"❌ Flashcard generation error: {e}")
            return self._generate_basic_flashcards(text, num_cards)
    def _generate_basic_flashcards(self, text: str, num_cards: int) -> List[Dict]:
        """Generate basic flashcards as fallback"""
        words = text.split()
        if len(words) < 50:
            return []
        # Extract key terms (simplified approach)
        sentences = text.split('.')[:num_cards]
        flashcards = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:
                flashcards.append({
                    'question': f"What is discussed about: {sentence.strip()[:50]}...?",
                    'answer': sentence.strip(),
                    'difficulty': 'Basic',
                    'category': 'Document Content',
                    'hint': 'Review the document content'
                })
        return flashcards[:num_cards]

class QuizAgent:
    def __init__(self, client: GroqClient):
        self.client = client
    def generate_quiz_structured(self, text: str, num_questions=8) -> List[Dict]:
        """Generate structured quiz data for the app interface"""
        if not text.strip():
            return []
        if len(text.split()) < 30:
            return []
        # Truncate text if too long
        max_chars = MAX_QUIZ_CHARS
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        prompt = f"""Create {num_questions} multiple choice questions based on the following content. Return ONLY a valid JSON array with no additional text.
Content:
{text}
Return format (MUST be valid JSON):
[
  {{
      "question": "Clear, specific question",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Detailed explanation of the correct answer",
      "difficulty": "Basic"
  }},
  {{
      "question": "Another question",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "correct_answer": 2,
      "explanation": "Another explanation",
      "difficulty": "Intermediate"
  }}
]
Guidelines:
- Test understanding and application, not just memorization
- Make all options plausible but only one clearly correct
- Mix difficulty: Basic, Intermediate, Advanced
- correct_answer should be the index (0-3) of the correct option
- Provide educational explanations"""
        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=3000)
            # Clean response to extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            # Try to parse JSON response
            try:
                quiz_data = json.loads(response)
                if isinstance(quiz_data, list) and len(quiz_data) > 0:
                    # Validate structure
                    valid_questions = []
                    for q in quiz_data:
                        if (isinstance(q, dict) and 'question' in q and 'options' in q
                            and 'correct_answer' in q and isinstance(q['options'], list)
                            and len(q['options']) == 4):
                            # Ensure valid correct_answer index
                            correct_idx = q.get('correct_answer', 0)
                            if not isinstance(correct_idx, int) or correct_idx < 0 or correct_idx > 3:
                                correct_idx = 0
                            valid_question = {
                                'question': str(q.get('question', '')),
                                'options': [str(opt) for opt in q['options'][:4]],
                                'correct_answer': correct_idx,
                                'explanation': str(q.get('explanation', 'No explanation provided')),
                                'difficulty': q.get('difficulty', 'Basic')
                            }
                            valid_questions.append(valid_question)
                    if valid_questions:
                        print(f"✅ Generated {len(valid_questions)} quiz questions")
                        return valid_questions
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"Response: {response[:200]}...")
            # Fallback: generate basic quiz
            return self._generate_basic_quiz(text, num_questions)
        except Exception as e:
            print(f"❌ Quiz generation error: {e}")
            return self._generate_basic_quiz(text, num_questions)
    def _generate_basic_quiz(self, text: str, num_questions: int) -> List[Dict]:
        """Generate basic quiz as fallback"""
        words = text.split()
        if len(words) < 100:
            return []
        # Extract key sentences for questions
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30][:num_questions]
        quiz_questions = []
        for i, sentence in enumerate(sentences):
            if len(sentence) > 30:
                quiz_questions.append({
                    'question': f"According to the document, what is mentioned about the topic?",
                    'options': [
                        sentence[:50] + "..." if len(sentence) > 50 else sentence,
                        "This is not mentioned in the document",
                        "The document states the opposite",
                        "This is only partially correct"
                    ],
                    'correct_answer': 0,
                    'explanation': f"The document states: {sentence}",
                    'difficulty': 'Basic'
                })
        return quiz_questions[:num_questions]

##### REAL WEB DISCOVERY AGENTS #####
class EnhancedResearchDiscoveryAgent:
    def __init__(self, client):
        self.client = client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    def extract_smart_keywords_and_topic(self, text: str) -> Tuple[str, List[str], List[str]]:
        """Enhanced keyword extraction with academic focus"""
        if not text.strip():
            return "Academic Study Material", ["study", "learning"], []
        # Limit text for analysis
        analysis_text = text[:MAX_RESEARCH_KEYWORDS_CHARS]
        prompt = f"""Analyze this academic document and extract research-relevant information.
Document Content:
{analysis_text}
Extract and return ONLY a JSON object with this exact format:
{{
"main_topic": "Specific academic field/subject",
"research_keywords": ["technical_term1", "technical_term2", "concept1", "method1", "theory1"],
"broader_keywords": ["field1", "discipline1", "area1"],
"key_concepts": ["definition1", "principle1", "framework1"],
"academic_level": "undergraduate|graduate|research"
}}
Focus on:
- Technical terminology and jargon
- Research methodologies mentioned
- Theoretical frameworks
- Specific academic concepts
- Field-specific terms
- Author names or seminal works if mentioned
Avoid generic terms like "study", "research", "analysis" unless they're part of a specific methodology."""
        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=400)
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            # Parse JSON response
            data = json.loads(response)
            main_topic = data.get("main_topic", "Academic Study Material")
            research_keywords = data.get("research_keywords", [])
            broader_keywords = data.get("broader_keywords", [])
            key_concepts = data.get("key_concepts", [])
            # Combine and prioritize keywords
            all_keywords = research_keywords + key_concepts + broader_keywords
            print(f"✅ Enhanced extraction - Topic: {main_topic}")
            print(f"✅ Research keywords: {research_keywords}")
            print(f"✅ Key concepts: {key_concepts}")
            return main_topic, research_keywords[:6], all_keywords[:10]
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed in keyword extraction: {e}")
            return self._fallback_keyword_extraction(analysis_text)
        except Exception as e:
            print(f"❌ Error in keyword extraction: {e}")
            return self._fallback_keyword_extraction(analysis_text)
    def _fallback_keyword_extraction(self, text: str) -> Tuple[str, List[str], List[str]]:
        """Fallback keyword extraction using text analysis"""
        # Extract potential academic terms (capitalized words, technical terms)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Remove common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'From', 'As', 'An', 'A', 'All', 'Any', 'Both', 'Each', 'Few', 'More', 'Most', 'Other', 'Some', 'Such', 'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very'}
        filtered_words = [word for word in words if word not in common_words and len(word) > 3]
        # Count frequency and take most common
        word_counts = Counter(filtered_words)
        top_keywords = [word.lower() for word, count in word_counts.most_common(8)]
        # Try to determine topic from first few sentences
        sentences = text.split('.')[:3]
        topic = "Academic Study Material"
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                # Extract potential topic from first meaningful sentence
                potential_topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', sentence)
                if potential_topics:
                    topic = ' '.join(potential_topics[:2])
                    break
        return topic, top_keywords[:6], top_keywords
    def find_papers(self, text: str, max_papers: int = 8) -> List[Dict]:
        """Enhanced paper finding with smarter keyword extraction"""
        if not text.strip():
            return []
        # Extract enhanced keywords
        topic, research_keywords, all_keywords = self.extract_smart_keywords_and_topic(text)
        papers = []
        # Try multiple search strategies
        search_strategies = [
            ("specific", research_keywords[:3]),  # Most specific terms first
            ("conceptual", all_keywords[3:6]),    # Broader concepts
            ("combined", [topic.split()[0]] + research_keywords[:2])  # Topic + key terms
        ]
        for strategy_name, keywords in search_strategies:
            if not keywords:
                continue
            print(f"🔍 Strategy '{strategy_name}' with keywords: {keywords}")
            # Try arXiv
            try:
                arxiv_papers = self._search_arxiv_enhanced(keywords, max_papers//3)
                papers.extend(arxiv_papers)
            except Exception as e:
                print(f"❌ arXiv search failed for {strategy_name}: {e}")
            # Try Semantic Scholar
            try:
                semantic_papers = self._search_semantic_scholar_enhanced(keywords, max_papers//3)
                papers.extend(semantic_papers)
            except Exception as e:
                print(f"❌ Semantic Scholar search failed for {strategy_name}: {e}")
            # Try PubMed for life sciences
            if self._is_life_sciences_topic(topic, all_keywords):
                try:
                    pubmed_papers = self._search_pubmed_enhanced(keywords, max_papers//4)
                    papers.extend(pubmed_papers)
                except Exception as e:
                    print(f"❌ PubMed search failed for {strategy_name}: {e}")
            time.sleep(1)  # Rate limiting between strategies
        # Remove duplicates and rank by relevance
        unique_papers = self._deduplicate_and_rank_papers(papers, research_keywords + all_keywords)
        return unique_papers[:max_papers]
    def _is_life_sciences_topic(self, topic: str, keywords: List[str]) -> bool:
        """Check if topic is related to life sciences"""
        life_science_terms = {
            'biology', 'medical', 'medicine', 'health', 'clinical', 'biochemistry',
            'genetics', 'molecular', 'cell', 'protein', 'gene', 'dna', 'rna',
            'pharmaceutical', 'drug', 'therapy', 'treatment', 'disease', 'cancer',
            'neuroscience', 'psychology', 'physiology', 'anatomy', 'pathology',
            'microbiology', 'immunology', 'pharmacology', 'epidemiology'
        }
        topic_lower = topic.lower()
        keywords_lower = [k.lower() for k in keywords]
        return any(term in topic_lower for term in life_science_terms) or \
            any(any(term in keyword for term in life_science_terms) for keyword in keywords_lower)
    def _search_arxiv_enhanced(self, keywords: List[str], max_results: int) -> List[Dict]:
        """Enhanced arXiv search with better query construction"""
        if not keywords:
            return []
        # Create more sophisticated query
        # Use AND for precise terms, quotes for phrases
        query_parts = []
        for keyword in keywords[:3]:  # Limit to avoid complex queries
            if ' ' in keyword:
                query_parts.append(f'"{keyword}"')  # Phrase search
            else:
                query_parts.append(keyword)
        query = ' AND '.join(query_parts)
        url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                try:
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    # Extract categories for relevance scoring
                    categories = []
                    for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                        term = category.get('term', '')
                        if term:
                            categories.append(term)
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    if title_elem is not None and summary_elem is not None:
                        title = title_elem.text.strip()
                        abstract = summary_elem.text.strip()
                        # Calculate relevance score
                        relevance_score = self._calculate_relevance_score(
                            title, abstract, keywords
                        )
                        year = "2024"
                        if published_elem is not None:
                            try:
                                year = published_elem.text[:4]
                            except:
                                pass
                        paper = {
                            'title': title,
                            'authors': ', '.join(authors[:4]) if authors else 'Unknown Authors',
                            'year': year,
                            'source': 'arXiv',
                            'abstract': (abstract[:400] + "...") if len(abstract) > 400 else abstract,
                            'url': id_elem.text if id_elem is not None else '#',
                            'relevance_score': relevance_score,
                            'categories': categories[:3],
                            '_search_keywords': keywords
                        }
                        papers.append(paper)
                except Exception as e:
                    continue
            print(f"✅ Found {len(papers)} papers from arXiv")
            return papers
        except Exception as e:
            print(f"❌ arXiv search error: {e}")
            return []
    def _search_semantic_scholar_enhanced(self, keywords: List[str], max_results: int) -> List[Dict]:
        """Enhanced Semantic Scholar search"""
        if not keywords:
            return []
        # Create query with academic focus
        query = ' '.join(keywords[:3])
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(query)}&limit={max_results}&fields=title,authors,year,abstract,url,venue,citationCount,fieldsOfStudy"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                papers = []
                for paper_data in data.get('data', []):
                    try:
                        title = paper_data.get('title', 'Untitled')
                        abstract = paper_data.get('abstract', '')
                        # Calculate relevance score
                        relevance_score = self._calculate_relevance_score(
                            title, abstract, keywords
                        )
                        # Skip papers with very low relevance
                        if relevance_score < 0.3:
                            continue
                        authors_list = []
                        if paper_data.get('authors'):
                            authors_list = [author.get('name', '') for author in paper_data['authors'][:4]]
                        # Get fields of study for additional context
                        fields = paper_data.get('fieldsOfStudy', [])
                        paper = {
                            'title': title,
                            'authors': ', '.join(authors_list) if authors_list else 'Unknown Authors',
                            'year': str(paper_data.get('year', '2024')),
                            'source': paper_data.get('venue', 'Semantic Scholar'),
                            'abstract': (abstract[:400] + "...") if abstract and len(abstract) > 400 else abstract or 'No abstract available',
                            'url': paper_data.get('url', '#'),
                            'relevance_score': relevance_score,
                            'citation_count': paper_data.get('citationCount', 0),
                            'fields_of_study': fields[:3],
                            '_search_keywords': keywords
                        }
                        papers.append(paper)
                    except Exception as e:
                        continue
                print(f"✅ Found {len(papers)} relevant papers from Semantic Scholar")
                return papers
        except Exception as e:
            print(f"❌ Semantic Scholar search error: {e}")
        return []
    def _search_pubmed_enhanced(self, keywords: List[str], max_results: int) -> List[Dict]:
        """Enhanced PubMed search for life sciences"""
        if not keywords:
            return []
        # Create medical/biological query
        query_parts = []
        for keyword in keywords[:3]:
            if ' ' in keyword:
                query_parts.append(f'"{keyword}"[Title/Abstract]')
            else:
                query_parts.append(f'{keyword}[Title/Abstract]')
        query = ' AND '.join(query_parts)
        # Search for PMIDs
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={quote_plus(query)}&retmax={max_results}&retmode=json&sort=relevance"
        try:
            response = self.session.get(search_url, timeout=15)
            if response.status_code == 200:
                search_data = response.json()
                pmids = search_data.get('esearchresult', {}).get('idlist', [])
                if pmids:
                    # Get paper details
                    pmids_str = ','.join(pmids[:max_results])
                    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmids_str}&retmode=xml"
                    fetch_response = self.session.get(fetch_url, timeout=20)
                    if fetch_response.status_code == 200:
                        return self._parse_pubmed_xml_enhanced(fetch_response.content, keywords)
        except Exception as e:
            print(f"❌ PubMed search error: {e}")
        return []
    def _parse_pubmed_xml_enhanced(self, xml_content: bytes, keywords: List[str]) -> List[Dict]:
        """Enhanced PubMed XML parsing with relevance scoring"""
        papers = []
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else 'Untitled'
                    # Extract abstract
                    abstract_texts = []
                    for abstract_elem in article.findall('.//Abstract/AbstractText'):
                        if abstract_elem.text:
                            abstract_texts.append(abstract_elem.text)
                    abstract = ' '.join(abstract_texts) if abstract_texts else 'No abstract available'
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(title, abstract, keywords)
                    # Skip irrelevant papers
                    if relevance_score < 0.2:
                        continue
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        fname = author.find('ForeName')
                        lname = author.find('LastName')
                        if fname is not None and lname is not None:
                            authors.append(f"{fname.text} {lname.text}")
                    # Extract year
                    year_elem = article.find('.//PubDate/Year')
                    year = year_elem.text if year_elem is not None else '2024'
                    # Extract journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else 'PubMed'
                    # Extract PMID for URL
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ''
                    paper = {
                        'title': title,
                        'authors': ', '.join(authors[:4]) if authors else 'Unknown Authors',
                        'year': year,
                        'source': journal,
                        'abstract': (abstract[:400] + "...") if len(abstract) > 400 else abstract,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else '#',
                        'relevance_score': relevance_score,
                        'pmid': pmid,
                        '_search_keywords': keywords
                    }
                    papers.append(paper)
                except Exception as e:
                    continue
        except Exception as e:
            print(f"❌ Error parsing PubMed XML: {e}")
        print(f"✅ Found {len(papers)} relevant papers from PubMed")
        return papers
    def _calculate_relevance_score(self, title: str, abstract: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matching"""
        if not keywords:
            return 0.5  # Default score
        text_combined = (title + " " + abstract).lower()
        # Score calculation
        score = 0
        total_weight = 0
        for i, keyword in enumerate(keywords):
            keyword_lower = keyword.lower()
            weight = 1.0 - (i * 0.15)  # Decrease weight for later keywords
            total_weight += weight
            # Exact match in title (high score)
            if keyword_lower in title.lower():
                score += weight * 2.0
            # Exact match in abstract
            elif keyword_lower in abstract.lower():
                score += weight * 1.5
            # Partial match (for compound terms)
            elif any(part in text_combined for part in keyword_lower.split() if len(part) > 3):
                score += weight * 0.8
        # Normalize score
        if total_weight > 0:
            normalized_score = min(score / total_weight, 1.0)
        else:
            normalized_score = 0.0
        return normalized_score
    def _deduplicate_and_rank_papers(self, papers: List[Dict], all_keywords: List[str]) -> List[Dict]:
        """Remove duplicates and rank papers by relevance"""
        if not papers:
            return []
        # Remove duplicates based on title similarity
        unique_papers = []
        seen_titles = set()
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            # Create a normalized title for deduplication
            title_words = set(title.split())
            # Check if we've seen a very similar title
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                # Calculate Jaccard similarity
                if title_words and seen_words:
                    intersection = len(title_words.intersection(seen_words))
                    union = len(title_words.union(seen_words))
                    similarity = intersection / union if union > 0 else 0
                    if similarity > 0.8:  # 80% similarity threshold
                        is_duplicate = True
                        break
            if not is_duplicate and title:
                seen_titles.add(title)
                unique_papers.append(paper)
        # Rank by relevance score and other factors
        def ranking_score(paper):
            base_score = paper.get('relevance_score', 0.5)
            # Boost recent papers slightly
            try:
                year = int(paper.get('year', '2020'))
                recency_boost = min((year - 2020) * 0.05, 0.2) if year >= 2020 else 0
            except:
                recency_boost = 0
            # Boost papers with citation count (if available)
            citation_boost = 0
            if 'citation_count' in paper:
                citations = paper['citation_count']
                if citations > 100:
                    citation_boost = 0.1
                elif citations > 50:
                    citation_boost = 0.05
            return base_score + recency_boost + citation_boost
        # Sort by rank score (descending)
        unique_papers.sort(key=ranking_score, reverse=True)
        # Add readable relevance labels
        for paper in unique_papers:
            score = paper.get('relevance_score', 0.5)
            if score >= 0.8:
                paper['relevance_label'] = 'Highly Relevant'
            elif score >= 0.6:
                paper['relevance_label'] = 'Very Relevant'
            elif score >= 0.4:
                paper['relevance_label'] = 'Relevant'
            else:
                paper['relevance_label'] = 'Somewhat Relevant'
        print(f"✅ Deduplicated to {len(unique_papers)} unique papers, ranked by relevance")
        return unique_papers

class YouTubeDiscoveryAgent:
    def __init__(self, client):
        self.client = client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    def find_videos(self, keywords: List[str], topic: str, max_videos: int = 10) -> List[Dict]:
        """Find actual educational YouTube videos"""
        if not keywords:
            return []
        all_videos = []
        # Create diverse search queries
        search_queries = self._create_search_queries(keywords, topic)
        for query in search_queries[:3]:  # Limit queries to avoid rate limiting
            try:
                print(f"🔍 Searching YouTube for: '{query}'")
                videos = self._search_youtube_real(query, max_videos // 3 + 2)
                all_videos.extend(videos)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"❌ Search failed for '{query}': {e}")
                continue
        # Remove duplicates and rank by educational value
        unique_videos = self._deduplicate_and_rank_videos(all_videos, keywords)
        return unique_videos[:max_videos]
    def _create_search_queries(self, keywords: List[str], topic: str) -> List[str]:
        """Create diverse search queries for better results"""
        queries = []
        # Primary queries
        if topic:
            queries.extend([
                f"{topic} tutorial",
                f"{topic} explained",
                f"learn {topic}",
                f"{topic} course"
            ])
        # Keyword-based queries
        if keywords:
            key_terms = ' '.join(keywords[:2])
            queries.extend([
                f"{key_terms} tutorial",
                f"{key_terms} explained",
                f"how to {key_terms}",
                f"{key_terms} lecture"
            ])
        # Educational channel specific queries
        if keywords:
            queries.extend([
                f"{keywords[0]} Khan Academy",
                f"{keywords[0]} Crash Course",
                f"{keywords[0]} MIT"
            ])
        return queries
    def _search_youtube_real(self, query: str, max_results: int) -> List[Dict]:
        """Search YouTube and extract real video data"""
        try:
            # Search YouTube
            search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            # Extract video data from page
            videos = self._extract_video_data_from_html(response.text, max_results)
            if videos:
                print(f"✅ Found {len(videos)} videos from YouTube search")
                return videos
            else:
                print("⚠️ No videos extracted from YouTube search")
                return []
        except Exception as e:
            print(f"❌ YouTube search error: {e}")
            return []
    def _extract_video_data_from_html(self, html_content: str, max_results: int) -> List[Dict]:
        """Extract video data from YouTube HTML"""
        videos = []
        try:
            # Method 1: Extract from ytInitialData
            videos_from_script = self._extract_from_script_tags(html_content, max_results)
            if videos_from_script:
                return videos_from_script
            # Method 2: Extract from HTML elements (fallback)
            soup = BeautifulSoup(html_content, 'html.parser')
            # Look for video containers
            video_elements = soup.find_all('div', {'class': re.compile(r'ytd-video-renderer|ytd-compact-video-renderer')})
            for element in video_elements[:max_results]:
                try:
                    video_data = self._parse_video_element(element)
                    if video_data:
                        videos.append(video_data)
                except Exception as e:
                    continue
            return videos
        except Exception as e:
            print(f"❌ Error extracting video data: {e}")
            return []
    def _extract_from_script_tags(self, html_content: str, max_results: int) -> List[Dict]:
        """Extract video data from YouTube's JavaScript data"""
        try:
            # Find ytInitialData
            script_pattern = r'var ytInitialData = ({.+?});'
            match = re.search(script_pattern, html_content)
            if not match:
                return []
            data_str = match.group(1)
            data = json.loads(data_str)
            videos = []
            # Navigate through YouTube's data structure
            contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
            for section in contents:
                items = section.get('itemSectionRenderer', {}).get('contents', [])
                for item in items:
                    if 'videoRenderer' in item:
                        video_data = self._parse_video_renderer(item['videoRenderer'])
                        if video_data:
                            videos.append(video_data)
                            if len(videos) >= max_results:
                                return videos
            return videos
        except Exception as e:
            print(f"❌ Script extraction error: {e}")
            return []
    def _parse_video_renderer(self, video_renderer: dict) -> Dict:
        """Parse video data from YouTube's videoRenderer object"""
        try:
            video_id = video_renderer.get('videoId', '')
            if not video_id:
                return None
            # Extract title
            title_runs = video_renderer.get('title', {}).get('runs', [])
            title = ''.join([run.get('text', '') for run in title_runs]) if title_runs else 'Unknown Title'
            # Extract channel name
            owner_text = video_renderer.get('ownerText', {}).get('runs', [])
            channel = owner_text[0].get('text', 'Unknown Channel') if owner_text else 'Unknown Channel'
            # Extract view count
            view_count_text = video_renderer.get('viewCountText', {}).get('simpleText', 'Unknown views')
            # Extract duration
            duration_text = video_renderer.get('lengthText', {}).get('simpleText', 'Unknown')
            # Extract description/snippet
            description_snippets = video_renderer.get('detailedMetadataSnippets', [])
            description = ''
            if description_snippets:
                snippet_runs = description_snippets[0].get('snippetText', {}).get('runs', [])
                description = ''.join([run.get('text', '') for run in snippet_runs])
            # Calculate educational score
            educational_score = self._calculate_educational_score(title, channel, description)
            return {
                'title': title[:100] + ('...' if len(title) > 100 else ''),
                'channel': channel,
                'duration': duration_text,
                'views': view_count_text,
                'description': description[:200] + ('...' if len(description) > 200 else ''),
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'educational_score': educational_score,
                'video_id': video_id
            }
        except Exception as e:
            print(f"❌ Error parsing video renderer: {e}")
            return None
    def _parse_video_element(self, element) -> Dict:
        """Parse video data from HTML element (fallback method)"""
        try:
            # This is a simplified fallback - YouTube's structure changes frequently
            # In practice, you'd need to adapt to current HTML structure
            return None
        except:
            return None
    def _calculate_educational_score(self, title: str, channel: str, description: str) -> str:
        """Calculate educational value score"""
        score = 0
        # Educational channels get high scores
        educational_channels = [
            'khan academy', 'crash course', 'ted-ed', 'mit opencourseware',
            '3blue1brown', 'veritasium', 'scishow', 'minutephysics',
            'professor leonard', 'patrickjmt', 'organic chemistry tutor',
            'bozeman science', 'amoeba sisters', 'crash course'
        ]
        channel_lower = channel.lower()
        if any(edu_channel in channel_lower for edu_channel in educational_channels):
            score += 3
        # Educational keywords in title
        educational_words = [
            'tutorial', 'explained', 'how to', 'course', 'lecture', 'lesson',
            'guide', 'introduction', 'basics', 'fundamentals', 'learn'
        ]
        title_lower = title.lower()
        for word in educational_words:
            if word in title_lower:
                score += 1
        # Length suggests educational content
        if len(description) > 50:
            score += 1
        # Return score category
        if score >= 4:
            return 'Excellent'
        elif score >= 3:
            return 'Very Good'
        elif score >= 2:
            return 'Good'
        else:
            return 'Fair'
    def _deduplicate_and_rank_videos(self, videos: List[Dict], keywords: List[str]) -> List[Dict]:
        """Remove duplicates and rank videos by educational value"""
        if not videos:
            return []
        # Remove duplicates based on video ID or URL
        seen_ids = set()
        unique_videos = []
        for video in videos:
            video_id = video.get('video_id') or video.get('url', '').split('v=')[-1].split('&')[0]
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                unique_videos.append(video)
        # Rank by educational score and relevance
        def rank_score(video):
            base_score = 0
            # Educational score ranking
            edu_score = video.get('educational_score', 'Fair')
            if edu_score == 'Excellent':
                base_score += 4
            elif edu_score == 'Very Good':
                base_score += 3
            elif edu_score == 'Good':
                base_score += 2
            else:
                base_score += 1
            # Keyword relevance in title
            title = video.get('title', '').lower()
            for keyword in keywords:
                if keyword.lower() in title:
                    base_score += 0.5
            return base_score
        # Sort by rank score
        unique_videos.sort(key=rank_score, reverse=True)
        print(f"✅ Ranked {len(unique_videos)} unique educational videos")
        return unique_videos

class WebResourceAgent:
    def __init__(self, client):
        self.client = client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive'
        })
    def find_resources(self, keywords: List[str], topic: str, max_resources: int = 12) -> List[Dict]:
        """Find actual web learning resources"""
        if not keywords:
            return []
        all_resources = []
        # Search different types of educational platforms
        search_strategies = [
            ('wikipedia', self._search_wikipedia),
            ('coursera', self._search_coursera),
            ('edx', self._search_edx),
            ('mit_ocw', self._search_mit_ocw),
            ('khan_academy', self._search_khan_academy),
            ('documentation', self._search_documentation_sites),
            ('general_search', self._search_general_educational)
        ]
        for strategy_name, search_func in search_strategies:
            try:
                print(f"🔍 Searching {strategy_name} for resources...")
                resources = search_func(keywords, topic, max_resources // len(search_strategies) + 1)
                all_resources.extend(resources)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"❌ {strategy_name} search failed: {e}")
                continue
        # Remove duplicates and rank
        unique_resources = self._deduplicate_and_rank_resources(all_resources, keywords)
        return unique_resources[:max_resources]
    def _search_wikipedia(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search Wikipedia for relevant articles"""
        resources = []
        # Search queries
        queries = [topic] + keywords[:2] if topic else keywords[:3]
        for query in queries:
            try:
                # Search Wikipedia
                search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
                response = self.session.get(search_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Skip disambiguation pages
                    if data.get('type') == 'disambiguation':
                        continue
                    resource = {
                        'title': data.get('title', query),
                        'type': 'Reference Article',
                        'source': 'Wikipedia',
                        'description': data.get('extract', '')[:300] + ('...' if len(data.get('extract', '')) > 300 else ''),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', f"https://en.wikipedia.org/wiki/{quote_plus(query)}"),
                        'quality_score': 'High',
                        'category': 'Reference'
                    }
                    resources.append(resource)
                elif response.status_code == 404:
                    # Try search API for similar articles
                    search_api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
                    continue
            except Exception as e:
                print(f"❌ Wikipedia search error for '{query}': {e}")
                continue
        return resources[:max_results]
    def _search_coursera(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search Coursera for courses"""
        resources = []
        try:
            # Create search query
            query = topic if topic else ' '.join(keywords[:2])
            search_url = f"https://www.coursera.org/search?query={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for course cards
                course_cards = soup.find_all('div', {'data-testid': 'search-card'}) or \
                               soup.find_all('div', class_=re.compile(r'course-card|search-result'))
                for card in course_cards[:max_results]:
                    try:
                        # Extract course information
                        title_elem = card.find('h3') or card.find('h2') or card.find('a')
                        title = title_elem.get_text(strip=True) if title_elem else 'Coursera Course'
                        # Get link
                        link_elem = card.find('a', href=True)
                        url = urljoin('https://www.coursera.org', link_elem['href']) if link_elem else 'https://www.coursera.org'
                        # Get description
                        desc_elem = card.find('p') or card.find('div', class_=re.compile(r'description'))
                        description = desc_elem.get_text(strip=True)[:200] if desc_elem else f"Coursera course on {query}"
                        resource = {
                            'title': title,
                            'type': 'Online Course',
                            'source': 'Coursera',
                            'description': description + ('...' if len(description) == 200 else ''),
                            'url': url,
                            'quality_score': 'High',
                            'category': 'Course'
                        }
                        resources.append(resource)
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"❌ Coursera search error: {e}")
        return resources
    def _search_edx(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search edX for courses"""
        resources = []
        try:
            query = topic if topic else ' '.join(keywords[:2])
            search_url = f"https://www.edx.org/search?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for course listings
                course_elements = soup.find_all('div', class_=re.compile(r'course-card|discovery-card'))
                for element in course_elements[:max_results]:
                    try:
                        title_elem = element.find('h3') or element.find('h2')
                        title = title_elem.get_text(strip=True) if title_elem else 'edX Course'
                        link_elem = element.find('a', href=True)
                        url = urljoin('https://www.edx.org', link_elem['href']) if link_elem else 'https://www.edx.org'
                        desc_elem = element.find('p')
                        description = desc_elem.get_text(strip=True)[:200] if desc_elem else f"edX course on {query}"
                        resource = {
                            'title': title,
                            'type': 'Online Course',
                            'source': 'edX',
                            'description': description + ('...' if len(description) == 200 else ''),
                            'url': url,
                            'quality_score': 'High',
                            'category': 'Course'
                        }
                        resources.append(resource)
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"❌ edX search error: {e}")
        return resources
    def _search_mit_ocw(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search MIT OpenCourseWare"""
        resources = []
        try:
            query = topic if topic else ' '.join(keywords[:2])
            search_url = f"https://ocw.mit.edu/search/?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for course results
                course_links = soup.find_all('a', href=re.compile(r'/courses/'))
                seen_urls = set()
                for link in course_links[:max_results]:
                    try:
                        if link['href'] in seen_urls:
                            continue
                        seen_urls.add(link['href'])
                        title = link.get_text(strip=True)
                        if not title or len(title) < 10:
                            continue
                        url = urljoin('https://ocw.mit.edu', link['href'])
                        resource = {
                            'title': title,
                            'type': 'Course Materials',
                            'source': 'MIT OpenCourseWare',
                            'description': f"MIT course materials covering {title}. Includes lectures, assignments, and readings.",
                            'url': url,
                            'quality_score': 'Excellent',
                            'category': 'Academic Course'
                        }
                        resources.append(resource)
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"❌ MIT OCW search error: {e}")
        return resources
    def _search_khan_academy(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search Khan Academy"""
        resources = []
        try:
            query = topic if topic else ' '.join(keywords[:2])
            search_url = f"https://www.khanacademy.org/search?search_again=1&page_search_query={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for content links
                content_links = soup.find_all('a', href=True)
                seen_titles = set()
                for link in content_links[:max_results * 2]:  # Get more to filter
                    try:
                        href = link['href']
                        if not any(path in href for path in ['/math/', '/science/', '/computing/', '/humanities/', '/economics/']):
                            continue
                        title = link.get_text(strip=True)
                        if not title or len(title) < 5 or title in seen_titles:
                            continue
                        seen_titles.add(title)
                        url = urljoin('https://www.khanacademy.org', href)
                        resource = {
                            'title': title,
                            'type': 'Interactive Lessons',
                            'source': 'Khan Academy',
                            'description': f"Khan Academy interactive lessons and exercises on {title}. Includes videos, practice problems, and progress tracking.",
                            'url': url,
                            'quality_score': 'High',
                            'category': 'Interactive Learning'
                        }
                        resources.append(resource)
                        if len(resources) >= max_results:
                            break
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"❌ Khan Academy search error: {e}")
        return resources
    def _search_documentation_sites(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search technical documentation sites"""
        resources = []
        # Documentation sites to search
        doc_sites = [
            ('MDN Web Docs', 'https://developer.mozilla.org/en-US/search?q='),
            ('W3Schools', 'https://www.w3schools.com/tags/'),
            ('Python.org', 'https://docs.python.org/3/search.html?q='),
        ]
        # Check if keywords suggest technical content
        tech_keywords = ['programming', 'coding', 'javascript', 'python', 'html', 'css', 'web', 'api', 'database', 'algorithm']
        is_technical = any(any(tech in keyword.lower() for tech in tech_keywords) for keyword in keywords)
        if not is_technical and topic:
            is_technical = any(tech in topic.lower() for tech in tech_keywords)
        if is_technical:
            for site_name, base_url in doc_sites:
                try:
                    query = keywords[0] if keywords else topic
                    search_url = f"{base_url}{quote_plus(query)}"
                    # For simplicity, create relevant documentation resources
                    resource = {
                        'title': f"{query.title()} Documentation",
                        'type': 'Technical Documentation',
                        'source': site_name,
                        'description': f"Official documentation and tutorials for {query} from {site_name}. Includes examples, best practices, and reference materials.",
                        'url': search_url,
                        'quality_score': 'High',
                        'category': 'Documentation'
                    }
                    resources.append(resource)
                except Exception as e:
                    continue
        return resources[:max_results]
    def _search_general_educational(self, keywords: List[str], topic: str, max_results: int) -> List[Dict]:
        """Search general educational resources using DuckDuckGo"""
        resources = []
        try:
            # Create educational search query
            query_base = topic if topic else ' '.join(keywords[:2])
            educational_queries = [
                f"{query_base} tutorial",
                f"{query_base} guide",
                f"learn {query_base}",
                f"{query_base} course"
            ]
            for query in educational_queries[:2]:  # Limit queries
                try:
                    # Use DuckDuckGo for search (more permissive than Google)
                    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query + ' site:edu OR site:org')}"
                    response = self.session.get(search_url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract search results
                        result_links = soup.find_all('a', {'class': 'result__a'})
                        for link in result_links[:max_results]:
                            try:
                                title = link.get_text(strip=True)
                                url = link.get('href', '')
                                if not title or not url or len(title) < 10:
                                    continue
                                # Determine resource type based on URL
                                resource_type = self._determine_resource_type(url)
                                resource = {
                                    'title': title[:80] + ('...' if len(title) > 80 else ''),
                                    'type': resource_type,
                                    'source': urlparse(url).netloc,
                                    'description': f"Educational resource about {query_base}. {title}",
                                    'url': url,
                                    'quality_score': 'Good',
                                    'category': 'Educational Resource'
                                }
                                resources.append(resource)
                            except Exception as e:
                                continue
                except Exception as e:
                    print(f"❌ General search error for '{query}': {e}")
                    continue
        except Exception as e:
            print(f"❌ General educational search error: {e}")
        return resources
    def _determine_resource_type(self, url: str) -> str:
        """Determine resource type based on URL"""
        url_lower = url.lower()
        if 'edu' in url_lower:
            return 'Academic Resource'
        elif any(site in url_lower for site in ['coursera', 'edx', 'udemy']):
            return 'Online Course'
        elif any(site in url_lower for site in ['youtube', 'vimeo']):
            return 'Video Tutorial'
        elif any(site in url_lower for site in ['github', 'stackoverflow']):
            return 'Code Repository'
        elif 'wiki' in url_lower:
            return 'Wiki Article'
        elif any(term in url_lower for term in ['tutorial', 'guide', 'how-to']):
            return 'Tutorial'
        else:
            return 'Educational Resource'
    def _deduplicate_and_rank_resources(self, resources: List[Dict], keywords: List[str]) -> List[Dict]:
        """Remove duplicates and rank resources by quality and relevance"""
        if not resources:
            return []
        # Remove duplicates based on URL
        seen_urls = set()
        unique_resources = []
        for resource in resources:
            url = resource.get('url', '')
            # Normalize URL for comparison
            normalized_url = url.split('?')[0].lower().rstrip('/')
            if normalized_url and normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_resources.append(resource)
        # Rank resources
        def rank_score(resource):
            score = 0
            # Quality score
            quality = resource.get('quality_score', 'Good')
            if quality == 'Excellent':
                score += 4
            elif quality == 'High':
                score += 3
            elif quality == 'Good':
                score += 2
            else:
                score += 1
            # Source reputation
            source = resource.get('source', '').lower()
            if any(trusted in source for trusted in ['mit.edu', 'wikipedia', 'khan', 'coursera', 'edx']):
                score += 2
            elif any(edu in source for edu in ['.edu', '.org']):
                score += 1
            # Keyword relevance
            title = resource.get('title', '').lower()
            description = resource.get('description', '').lower()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in title:
                    score += 1
                elif keyword_lower in description:
                    score += 0.5
            return score
        # Sort by rank score
        unique_resources.sort(key=rank_score, reverse=True)
        print(f"✅ Ranked {len(unique_resources)} unique educational resources")
        return unique_resources

##### SLIDESMAKER AGENTS (NEWLY INTEGRATED) #####
@dataclass
class SlideContent:
    title: str
    content: List[str]  # Can be bullet points or paragraphs
    slide_type: str  # e.g., "title", "introduction", "content", "conclusion", "key_takeaways"
    notes: str = ""
    suggested_visuals: str = "" # New field for visual suggestions

@dataclass
class Presentation:
    title: str
    slides: List[SlideContent]
    theme: str = "professional"
    total_slides: int = 0

class Agent(ABC):
    def __init__(self, name: str, client: GroqClient):
        self.name = name
        self.client = client

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class OutlineAgent(Agent):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        topic = input_data.get("topic", "")
        audience = input_data.get("audience", "general")
        duration = input_data.get("duration", 10)
        document_text = input_data.get("document_text", "") # New: full document text

        # Truncate document_text if too long for outline generation
        if len(document_text) > MAX_PRESENTATION_CONTENT_CHARS:
            document_text = document_text[:MAX_PRESENTATION_CONTENT_CHARS] + "\n... [Document truncated for outline generation] ..."

        prompt = f"""Based on the following document content, create a detailed presentation outline.
Document Content:
{document_text}

Presentation Topic: "{topic}"
Target Audience: {audience}
Presentation Duration: Approximately {duration} minutes

Return ONLY a valid JSON object with the exact format below. Do not include any other text or markdown outside the JSON.

{{
"presentation_title": "A compelling and specific title for the presentation",
"slides_outline": [
  {{
    "slide_title": "Presentation Title",
    "slide_type": "title",
    "key_points_summary": "Main title, presenter name, date.",
    "estimated_duration_minutes": 0.5
  }},
  {{
    "slide_title": "Introduction: [Specific Subtopic]",
    "slide_type": "introduction",
    "key_points_summary": "Brief overview of the main topic, its relevance, and what the audience will learn. Refer to the document content.",
    "estimated_duration_minutes": 1
  }},
  // Add 4-7 more content slides based on key sections/concepts in the document
  {{
    "slide_title": "Key Takeaways",
    "slide_type": "key_takeaways",
    "key_points_summary": "Summarize the most crucial points and conclusions from the document.",
    "estimated_duration_minutes": 1
  }},
  {{
    "slide_title": "Q&A and Discussion",
    "slide_type": "conclusion",
    "key_points_summary": "Open floor for questions, provide contact info, and suggest next steps for learning.",
    "estimated_duration_minutes": 1
  }}
]
}}

Guidelines:
- Ensure the outline covers the most important aspects of the document.
- Each slide_title should be clear and descriptive.
- key_points_summary should be concise but informative, directly referencing the document's content.
- Mix slide types as appropriate (title, introduction, content, key_takeaways, conclusion).
- Aim for 6-10 slides in total, adjusting estimated_duration_minutes to fit the overall duration."""
        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=1000)
            parsed_data = self._parse_json_response(response)

            if parsed_data and "presentation_title" in parsed_data and "slides_outline" in parsed_data:
                print(f"✅ Outline generated with {len(parsed_data['slides_outline'])} slides.")
                return {
                    "outline": parsed_data["slides_outline"],
                    "title": parsed_data["presentation_title"],
                    "estimated_slides": len(parsed_data["slides_outline"])
                }
            else:
                print("❌ Failed to parse structured outline. Falling back to basic.")
                return self._fallback_outline(topic, duration)
        except Exception as e:
            print(f"❌ Error in OutlineAgent: {e}. Falling back to basic.")
            return self._fallback_outline(topic, duration)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Helper to parse JSON from LLM response, handling markdown code blocks."""
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Problematic JSON: {response_text[:500]}...")
            return {}

    def _fallback_outline(self, topic: str, duration: int) -> Dict[str, Any]:
        """Basic fallback outline if AI generation fails."""
        return {
            "outline": [
                {"slide_title": f"Presentation on {topic}", "slide_type": "title", "key_points_summary": "Main title, presenter name, date.", "estimated_duration_minutes": 0.5},
                {"slide_title": f"Introduction to {topic}", "slide_type": "introduction", "key_points_summary": "Overview of the topic.", "estimated_duration_minutes": 1},
                {"slide_title": f"Key Aspects of {topic}", "slide_type": "content", "key_points_summary": "Main points and details.", "estimated_duration_minutes": duration - 2},
                {"slide_title": "Conclusion", "slide_type": "conclusion", "key_points_summary": "Summary and next steps.", "estimated_duration_minutes": 1}
            ],
            "title": f"Presentation on {topic}",
            "estimated_slides": 4
        }

class ContentAgent(Agent):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        slide_outline = input_data.get("slide_outline", {})
        full_document_text = input_data.get("full_document_text", "") # New: full document text
        presentation_context = input_data.get("context", "")

        slide_title = slide_outline.get('slide_title', 'Untitled Slide')
        key_points_summary = slide_outline.get('key_points_summary', '')
        slide_type = slide_outline.get('slide_type', 'content')

        # Truncate full_document_text if too long for content generation
        if len(full_document_text) > MAX_PRESENTATION_CONTENT_CHARS:
            full_document_text = full_document_text[:MAX_PRESENTATION_CONTENT_CHARS] + "\n... [Document truncated for detailed content generation] ..."

        prompt = f"""Generate detailed content for a presentation slide based on the following information and document context.
Slide Title: "{slide_title}"
Key Points Summary for this slide: "{key_points_summary}"
Presentation Context: {presentation_context}

Document Content for reference:
{full_document_text}

Return ONLY a valid JSON object with the exact format below. Do not include any other text or markdown outside the JSON.

{{
"slide_title": "The exact slide title provided above",
"bullet_points": [
  "First key point, directly supported by the document content.",
  "Second key point, elaborating on details from the document.",
  "Third key point, providing examples or implications from the document."
],
"speaker_notes": "Detailed notes for the presenter, explaining each bullet point and providing additional context or examples from the document. Aim for 100-200 words.",
"suggested_visuals": "Brief description of a relevant image, chart, or diagram (e.g., 'Diagram illustrating the data flow', 'Image of a neural network', 'Chart showing market trends')."
}}

Guidelines:
- Ensure bullet points are concise, impactful, and directly derived from the document content.
- Speaker notes should be comprehensive and provide enough detail for a presenter to elaborate.
- Suggested visuals should be relevant to the slide's content.
- Aim for 3-5 bullet points.
- If the slide_type is 'title', 'introduction', 'key_takeaways', or 'conclusion', adjust content accordingly (e.g., for title slide, bullet_points can be presenter name, date; for conclusion, next steps, Q&A)."""
        try:
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=1500) # Increased max_tokens for detailed content
            parsed_data = self._parse_json_response(response)

            if parsed_data and "slide_title" in parsed_data and "bullet_points" in parsed_data:
                print(f"   ✅ Content generated for slide: {slide_title}")
                return {
                    "slide": SlideContent(
                        title=parsed_data.get('slide_title', slide_title),
                        content=parsed_data.get('bullet_points', []),
                        slide_type=slide_type,
                        notes=parsed_data.get('speaker_notes', ''),
                        suggested_visuals=parsed_data.get('suggested_visuals', '')
                    ),
                    "success": True
                }
            else:
                print(f"   ❌ Failed to parse structured content for slide: {slide_title}. Falling back to basic.")
                return {"slide": self._fallback_slide_content(slide_title, key_points_summary, slide_type), "success": False}
        except Exception as e:
            print(f"   ❌ Error in ContentAgent for slide {slide_title}: {e}. Falling back to basic.")
            return {"slide": self._fallback_slide_content(slide_title, key_points_summary, slide_type), "success": False}

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Helper to parse JSON from LLM response, handling markdown code blocks."""
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Problematic JSON: {response_text[:500]}...")
            return {}

    def _fallback_slide_content(self, title: str, summary: str, slide_type: str) -> SlideContent:
        """Basic fallback slide content if AI generation fails."""
        if slide_type == "title":
            return SlideContent(title=title, content=["Your Name", "Date"], slide_type="title")
        elif slide_type == "introduction":
            return SlideContent(title=title, content=[summary, "Key topics to be covered"], slide_type="introduction")
        elif slide_type == "conclusion":
            return SlideContent(title=title, content=["Summary of key points", "Q&A", "Thank You"], slide_type="conclusion")
        elif slide_type == "key_takeaways":
            return SlideContent(title=title, content=["Point 1", "Point 2", "Point 3"], slide_type="key_takeaways")
        else:
            return SlideContent(title=title, content=[summary, "Further details from document"], slide_type="content")

class DesignAgent(Agent):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        presentation = input_data.get("presentation")
        theme_preference = input_data.get("theme", "professional")
        prompt = f"""Suggest design improvements for a {len(presentation.slides)}-slide presentation titled "{presentation.title}"
    Current theme preference: {theme_preference}
        Please provide:
    1. Color scheme recommendations
    2. Font suggestions
    3. Layout recommendations for different slide types
    4. Visual element suggestions
    5. Consistency guidelines
        Focus on professional, clean design."""
        design_text = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=500)
        design_suggestions = self._parse_design_suggestions(design_text)
        return {
            "design_guidelines": design_suggestions,
            "theme": theme_preference,
            "success": True
        }
    def _parse_design_suggestions(self, text: str) -> Dict[str, Any]:
        """Parse design suggestions into structured format"""
        # This is a simplified parsing, in a real app you'd use more robust methods
        return {
            "color_scheme": ["#1e3a8a", "#3b82f6", "#60a5fa", "#ffffff", "#f8fafc"],
            "fonts": {
                "heading": "Inter, Arial, sans-serif",
                "body": "Inter, Arial, sans-serif"
            },
            "layout": {
                "title_slide": "centered",
                "content_slide": "header_with_bullets",
                "conclusion_slide": "centered"
            },
            "suggestions": text
        }

class QualityAgent(Agent):
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        presentation = input_data.get("presentation")
        issues = []
        suggestions = []
        # Check presentation structure
        if len(presentation.slides) < 3:
            issues.append("Presentation seems too short")
            suggestions.append("Consider adding more detailed content slides")
        if len(presentation.slides) > 20:
            issues.append("Presentation might be too long")
            suggestions.append("Consider condensing content or splitting into multiple presentations")
        # Check slide content
        for i, slide in enumerate(presentation.slides):
            if len(slide.content) > 7:
                issues.append(f"Slide {i+1} has too many bullet points")
                suggestions.append(f"Reduce bullet points on slide {i+1} to 5 or fewer")
            if not slide.title:
                issues.append(f"Slide {i+1} missing title")
                suggestions.append(f"Add a clear title to slide {i+1}")
        # Generate overall quality assessment
        prompt = f"""Review this presentation structure and provide quality feedback:
    Title: {presentation.title}
    Number of slides: {len(presentation.slides)}
    Slide titles: {[slide.title for slide in presentation.slides]}
        Assess:
    1. Content flow and logical progression
    2. Audience engagement potential
    3. Clarity and conciseness
    4. Professional quality
        Provide specific improvement suggestions."""
        quality_text = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=400)
        return {
            "issues": issues,
            "suggestions": suggestions,
            "quality_assessment": quality_text,
            "overall_score": max(1, 10 - len(issues)),
            "success": True
        }

class CoordinatorAgent:
    def __init__(self, client: GroqClient):
        self.client = client
        self.agents = {
            "outline": OutlineAgent("OutlineAgent", client),
            "content": ContentAgent("ContentAgent", client),
            "design": DesignAgent("DesignAgent", client),
            "quality": QualityAgent("QualityAgent", client)
        }

    def create_presentation(self, topic: str, document_text: str, audience: str = "general",
                            duration: int = 10, theme: str = "professional") -> tuple:
        """Orchestrate the multi-agent workflow to create a presentation from document text."""

        print(f"🚀 Starting presentation creation for: {topic} from document.")

        # Step 1: Create outline
        print("📋 Generating presentation outline...")
        outline_result = self.agents["outline"].execute({
            "topic": topic,
            "audience": audience,
            "duration": duration,
            "document_text": document_text # Pass document text to outline agent
        })

        presentation_title = outline_result["title"]
        sections = outline_result["outline"]

        if not sections:
            print("❌ No outline sections generated. Cannot create presentation.")
            return None, None, None

        print(f"✅ Outline complete: {len(sections)} sections")

        # Step 2: Generate content for each section
        print("📝 Generating slide content...")
        slides = []

        for i, section in enumerate(sections):
            print(f"   Creating slide {i+1}/{len(sections)}: {section.get('slide_title', 'Untitled')}...")
            content_result = self.agents["content"].execute({
                "slide_outline": section,
                "full_document_text": document_text, # Pass full document text to content agent
                "context": f"This is part of a presentation about {topic} for {audience}, based on the provided document."
            })
            slides.append(content_result["slide"])
            time.sleep(0.5)  # Rate limiting for API calls

        presentation = Presentation(
            title=presentation_title,
            slides=slides,
            theme=theme,
            total_slides=len(slides)
        )

        print("✅ Content generation complete")

        # Step 3: Apply design guidelines
        print("🎨 Applying design guidelines...")
        design_result = self.agents["design"].execute({
            "presentation": presentation,
            "theme": theme
        })

        print("✅ Design guidelines applied")

        # Step 4: Quality check
        print("🔍 Performing quality check...")
        quality_result = self.agents["quality"].execute({
            "presentation": presentation
        })

        print(f"✅ Quality check complete - Score: {quality_result['overall_score']}/10")

        if quality_result["issues"]:
            print("⚠️ Issues found:")
            for issue in quality_result["issues"]:
                print(f"   - {issue}")
            print("💡 Suggestions:")
            for suggestion in quality_result["suggestions"]:
                print(f"   - {suggestion}")

        return presentation, design_result["design_guidelines"], quality_result

class PresentationExporter:
    @staticmethod
    def to_json(presentation: Presentation, design_guidelines: Dict, quality_result: Dict) -> str:
        """Export presentation to JSON format"""
        export_data = {
            "presentation": {
                "title": presentation.title,
                "theme": presentation.theme,
                "total_slides": presentation.total_slides,
                "slides": [
                    {
                        "title": slide.title,
                        "content": slide.content,
                        "slide_type": slide.slide_type,
                        "notes": slide.notes,
                        "suggested_visuals": slide.suggested_visuals
                    }
                    for slide in presentation.slides
                ]
            },
            "design_guidelines": design_guidelines,
            "quality_assessment": quality_result
        }
        return json.dumps(export_data, indent=2)
    @staticmethod
    def to_markdown(presentation: Presentation) -> str:
        """Export presentation to Markdown format"""
        md_content = f"# {presentation.title}\n\n"
        for i, slide in enumerate(presentation.slides, 1):
            md_content += f"## Slide {i}: {slide.title}\n\n"
            if slide.slide_type == "title":
                md_content += f"**{slide.title}**\n\n"
                for item in slide.content:
                    md_content += f"*{item}*\n\n"
            else:
                for item in slide.content:
                    md_content += f"- {item}\n"
                md_content += "\n"
            if slide.notes:
                md_content += f"*Speaker Notes: {slide.notes}*\n\n"
            if slide.suggested_visuals:
                md_content += f"*Suggested Visuals: {slide.suggested_visuals}*\n\n"
            md_content += "---\n\n"
        return md_content

##### DIAGNOSTIC TOOLS #####
def diagnose_pdf(file_path):
    """Enhanced PDF diagnostic tool"""
    print(f"\n🔍 Diagnosing PDF: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    try:
        with pdfplumber.open(file_path) as pdf:
            print(f"📄 Pages: {len(pdf.pages)}")
            print(f"📋 Metadata: {pdf.metadata}")
            total_chars = 0
            text_pages = 0
            for i, page in enumerate(pdf.pages[:5], 1):  # Check first 5 pages
                print(f"\n--- Page {i} Analysis ---")
                # Try different extraction methods
                text_standard = page.extract_text() or ""
                text_layout = page.extract_text(layout=True) or ""
                chars_count = len(page.chars) if hasattr(page, 'chars') else 0
                images_count = len(page.images) if hasattr(page, 'images') else 0
                print(f"Standard extraction: {len(text_standard)} chars")
                print(f"Layout extraction: {len(text_layout)} chars")
                print(f"Page chars: {chars_count}")
                print(f"Images: {images_count}")
                if len(text_standard) > 20:
                    text_pages += 1
                    total_chars += len(text_standard)
                    print(f"✅ Text found: {text_standard[:100]}...")
                else:
                    print("❌ Little/no text extracted")
            print(f"\n📊 Summary:")
            print(f"Text-extractable pages: {text_pages}/{min(5, len(pdf.pages))}")
            print(f"Total characters: {total_chars}")
            print(f"Recommendation: {'Text extraction should work' if text_pages > 0 else 'May need OCR'}")
    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")

def test_ocr_setup():
    """Test OCR capabilities"""
    print("\n🔍 Testing OCR Setup...")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        # Create test image
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
            draw.text((10, 10), "OCR Test: Hello World! 123", fill='black', font=font)
        except:
            draw.text((10, 10), "OCR Test: Hello World! 123", fill='black')
        # Test OCR
        text = pytesseract.image_to_string(img)
        print(f"✅ OCR output: '{text.strip()}'")
        if "Hello World" in text or "OCR Test" in text:
            print("✅ OCR is working correctly!")
        else:
            print("⚠️ OCR may have issues with text recognition")
    except Exception as e:
        print(f"❌ OCR setup failed: {e}")
        print("💡 Try installing Tesseract: https://github.com/tesseract-ocr/tesseract")

def test_groq_connection():
    """Test Groq API connection"""
    print("\n🔍 Testing Groq Connection...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ No Groq API key found in environment variables")
        print("💡 Set GROQ_API_KEY in your .env file")
        return
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    try:
        client = GroqClient()
        test_message = [{"role": "user", "content": "Say 'Hello World' to test the connection."}]
        response = client.chat_completion(test_message, max_tokens=50)
        if "❌" in response:
            print(f"❌ Connection failed: {response}")
        else:
            print(f"✅ Connection successful: {response}")
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

##### MAIN EXECUTION LOGIC #####
def run_study_assistant(pdf_path):
    """Main function to process PDF and generate study materials"""
    print(f"\n🎓 Processing: {pdf_path}")
    try:
        # Initialize components
        client = GroqClient()
        processor = EnhancedPDFProcessor()
        summary_agent = SummaryAgent(client)
        flashcard_agent = FlashcardAgent(client)
        quiz_agent = QuizAgent(client)

        # Extract text
        print("\n📄 Extracting text from PDF...")
        result = processor.extract_text_with_ocr(pdf_path)
        if result["status"] == "error":
            print(f"❌ PDF Processing Failed: {result['message']}")
            return
        print(f"\n{result['message']}")

        extracted_text = result["text"]
        if result["word_count"] < 20:
            print("⚠️ Very little content extracted. Results may be limited.")

        # Generate study materials
        print("\n🧠 Generating study materials...")

        print("📘 Creating summary...")
        summary = summary_agent.generate_summary(extracted_text)

        print("🃏 Creating flashcards...")
        flashcards = flashcard_agent.generate_flashcards_structured(extracted_text)

        print("📝 Creating quiz...")
        quiz = quiz_agent.generate_quiz_structured(extracted_text)

        # Display results
        print("\n" + "="*80)
        print("📘 SUMMARY")
        print("="*80)
        print(summary)

        print("\n" + "="*80)
        print("🃏 FLASHCARDS")
        print("="*80)
        for i, card in enumerate(flashcards, 1):
            print(f"\nCard {i}:")
            print(f"Q: {card['question']}")
            print(f"A: {card['answer']}")
            print(f"Difficulty: {card['difficulty']}")
            print(f"Category: {card['category']}")
            if card.get('hint'):
                print(f"Hint: {card['hint']}")

        print("\n" + "="*80)
        print("📝 QUIZ")
        print("="*80)
        for i, q in enumerate(quiz, 1):
            print(f"\nQuestion {i}: {q['question']}")
            for j, option in enumerate(q['options']):
                print(f"  {chr(65+j)}) {option}")
            print(f"Correct: {chr(65+q['correct_answer'])}")
            print(f"Explanation: {q['explanation']}")

        # New: Presentation Generation
        print("\n" + "="*80)
        print("📊 GENERATING PRESENTATION")
        print("="*80)
        coordinator = CoordinatorAgent(client)
        presentation, design_guidelines, quality_result = coordinator.create_presentation(
            topic="Document Content Analysis", # Default topic, can be refined by user input
            document_text=extracted_text,
            audience="students",
            duration=15 # Default duration
        )

        if presentation:
            print("\n" + "="*80)
            print("✨ PRESENTATION GENERATED")
            print("="*80)
            print(f"Title: {presentation.title}")
            print(f"Total Slides: {presentation.total_slides}")
            print("\n--- Slides Overview ---")
            for i, slide in enumerate(presentation.slides, 1):
                print(f"Slide {i}: {slide.title} ({slide.slide_type})")
                if slide.suggested_visuals:
                    print(f"  Visuals: {slide.suggested_visuals}")

            print("\n--- Design Guidelines ---")
            print(json.dumps(design_guidelines, indent=2))

            print("\n--- Quality Assessment ---")
            print(quality_result["quality_assessment"])

            # Export to Markdown for easy viewing
            markdown_output = PresentationExporter.to_markdown(presentation)
            print("\n--- Markdown Presentation ---")
            print(markdown_output)

            # Optionally save to file
            # with open("generated_presentation.md", "w", encoding="utf-8") as f:
            #     f.write(markdown_output)
            # print("\nPresentation saved to generated_presentation.md")

    except Exception as e:
        print(f"❌ Critical error in run_study_assistant: {e}")

if __name__ == "__main__":
    import uvicorn
    print("\n🎓 AI Study Assistant CLI - Enhanced Version")
    print("="*50)
    print("1. Run Study Assistant (includes Presentation Generation)")
    print("2. Diagnose PDF")
    print("3. Test OCR Setup")
    print("4. Test Groq Connection")
    print("5. Exit")

    choice = input("\nChoose an option (1-5): ").strip()
    if choice == "1":
        pdf_path = input("Enter PDF file path: ").strip().strip('"')
        if os.path.exists(pdf_path):
            run_study_assistant(pdf_path)
        else:
            print(f"❌ File not found: {pdf_path}")

    elif choice == "2":
        pdf_path = input("Enter PDF file path: ").strip().strip('"')
        diagnose_pdf(pdf_path)

    elif choice == "3":
        test_ocr_setup()

    elif choice == "4":
        test_groq_connection()

    elif choice == "5":
        print("👋 Goodbye!")

    else:
        print("❌ Invalid option. Please choose 1-5.")
