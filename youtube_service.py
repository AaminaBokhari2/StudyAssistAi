import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()  # Load environment variables

class YouTubeService:
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YouTube API key not found in environment variables")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def search_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search YouTube for educational videos"""
        try:
            request = self.youtube.search().list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="video",
                videoDuration="medium",  # Filters out shorts
                relevanceLanguage="en",
                safeSearch="moderate"
            )
            response = request.execute()
            
            videos = []
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                snippet = item['snippet']
                videos.append({
                    'title': snippet['title'],
                    'channel': snippet['channelTitle'],
                    'description': snippet['description'][:150] + '...',
                    'url': f"https://youtu.be/{video_id}",
                    'published_at': snippet['publishedAt'],
                    'thumbnail': snippet['thumbnails']['default']['url']
                })
            
            return videos
            
        except Exception as e:
            print(f"YouTube API error: {e}")
            return self._get_fallback_videos(query)

    def _get_fallback_videos(self, query: str) -> List[Dict]:
        """Provide fallback content when API fails"""
        return [{
            'title': f"Educational video about {query}",
            'channel': "Fallback Channel",
            'description': "This would be a video about the topic",
            'url': "https://youtu.be/example",
            'published_at': "2023-01-01",
            'thumbnail': ""
        }]