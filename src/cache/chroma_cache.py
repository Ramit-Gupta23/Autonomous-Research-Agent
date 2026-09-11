import json
import chromadb
from typing import List, Optional


class ResearchCache:
    """
    ChromaDB mein research results store karta hai.
    Same topic dobara search ho toh API call nahi karta.
    """

    def __init__(self):
        # Local disk pe store hoga — ./chroma_db folder mein
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Do alag collections — web aur arxiv ke liye
        self.web_collection = self.client.get_or_create_collection(
            name="web_results",
            metadata={"description": "Tavily web search results"}
        )

        self.arxiv_collection = self.client.get_or_create_collection(
            name="arxiv_results",
            metadata={"description": "Arxiv paper results"}
        )

    def _clean_topic(self, topic: str) -> str:
        """Topic ko ID ke liye clean karo — spaces aur special chars hatao"""
        return topic.lower().strip().replace(" ", "_")[:50]

    # ─────────────────────────────
    # WEB RESULTS
    # ─────────────────────────────
    def get_web_results(self, topic: str) -> Optional[List[dict]]:
        """
        Cache mein web results hain toh return karo.
        Nahi hain toh None return karo.
        """
        try:
            topic_id = self._clean_topic(topic)
            result = self.web_collection.get(ids=[topic_id])

            if result["documents"] and result["documents"][0]:
                print(f"📦 Cache hit — web results for '{topic}'")
                # JSON string ko list mein convert karo
                return json.loads(result["metadatas"][0]["results"])

            return None  # Cache miss

        except Exception:
            return None  # Koi bhi error = cache miss

    def save_web_results(self, topic: str, results: List[dict]) -> None:
        """Web results ko ChromaDB mein save karo"""
        try:
            topic_id = self._clean_topic(topic)

            # Pehle delete karo agar exist karta hai
            try:
                self.web_collection.delete(ids=[topic_id])
            except Exception:
                pass

            self.web_collection.add(
                ids=[topic_id],
                documents=[topic],  # Topic as document
                metadatas=[{
                    "topic": topic,
                    "results": json.dumps(results),  # Results as JSON string
                    "count": str(len(results))
                }]
            )
            print(f"💾 Saved web results for '{topic}' to cache")

        except Exception as e:
            print(f"⚠️ Could not save web results to cache: {e}")

    # ─────────────────────────────
    # ARXIV RESULTS
    # ─────────────────────────────
    def get_arxiv_results(self, topic: str) -> Optional[List[dict]]:
        """Cache mein arxiv results hain toh return karo"""
        try:
            topic_id = self._clean_topic(topic)
            result = self.arxiv_collection.get(ids=[topic_id])

            if result["documents"] and result["documents"][0]:
                print(f"📦 Cache hit — arxiv results for '{topic}'")
                return json.loads(result["metadatas"][0]["results"])

            return None

        except Exception:
            return None

    def save_arxiv_results(self, topic: str, results: List[dict]) -> None:
        """Arxiv results ko ChromaDB mein save karo"""
        try:
            topic_id = self._clean_topic(topic)

            try:
                self.arxiv_collection.delete(ids=[topic_id])
            except Exception:
                pass

            self.arxiv_collection.add(
                ids=[topic_id],
                documents=[topic],
                metadatas=[{
                    "topic": topic,
                    "results": json.dumps(results),
                    "count": str(len(results))
                }]
            )
            print(f"💾 Saved arxiv results for '{topic}' to cache")

        except Exception as e:
            print(f"⚠️ Could not save arxiv results to cache: {e}")

    def is_cached(self, topic: str) -> bool:
        """Topic cached hai ya nahi — quick check"""
        return self.get_web_results(topic) is not None


# Global instance — poora project yahi use karega
cache = ResearchCache()