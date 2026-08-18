import unittest
from agent import initialize_node, route_retrieval
from vector_store import initialize_vector_store, check_city_in_store

class TestTravelAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = initialize_vector_store()

    def test_city_entity_extraction_lowercase(self):
        """
        Verify that a lowercase query like 'bangalore' has its city name 
        correctly title-cased and stored.
        """
        state = {"query": "tell me about bangalore", "messages": []}
        result = initialize_node(state)
        self.assertEqual(result["city"], "Bangalore")
        self.assertFalse(result["skip_summary"])

    def test_routing_local_db(self):
        """
        Verify that pre-populated cities (Tokyo, Paris, New York) route to local vector store.
        """
        # Test Tokyo
        state_tokyo = {"city": "Tokyo", "skip_summary": False}
        route = route_retrieval(state_tokyo)
        self.assertEqual(route, "vector_retrieve")
        
        # Test Paris
        state_paris = {"city": "Paris", "skip_summary": False}
        route = route_retrieval(state_paris)
        self.assertEqual(route, "vector_retrieve")

    def test_routing_web_search(self):
        """
        Verify that cities not in local store route to the web search fallback.
        """
        state_fallback = {"city": "Snohomish", "skip_summary": False}
        route = route_retrieval(state_fallback)
        self.assertEqual(route, "web_search")

    def test_routing_memory_bypass(self):
        """
        Verify that follow-up queries route to skip retrieval node.
        """
        state_followup = {"city": "Tokyo", "skip_summary": True}
        route = route_retrieval(state_followup)
        self.assertEqual(route, "skip_retrieval")

    def test_local_store_lookup(self):
        """
        Verify the custom local FAISS lookup returns matching facts.
        """
        is_present, fact = check_city_in_store("Tokyo", self.db)
        self.assertTrue(is_present)
        self.assertIn("capital city of Japan", fact)

        is_present_fallback, _ = check_city_in_store("Sydney", self.db)
        self.assertFalse(is_present_fallback)

if __name__ == "__main__":
    unittest.main()
