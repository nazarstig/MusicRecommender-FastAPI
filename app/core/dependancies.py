from functools import lru_cache

from app.services.recommendation import RecommendationService


@lru_cache
def _get_recommendation_singleton() -> RecommendationService:
    return RecommendationService()


def get_recommendation_service() -> RecommendationService:
    return _get_recommendation_singleton()