from pydantic import BaseModel, Field
from typing import List


class PlaylistRequest(BaseModel):
    """
    Schema for playlist-based recommendation request
    """
    track_ids: List[str] = Field(
        ..., 
        min_length=5, 
        max_length=5,
        description="Exactly 5 track IDs from the user's playlist"
    )


class RecommendationResponse(BaseModel):
    """
    Schema for a single song recommendation
    """
    TrackId: str
    TrackName: str
    ArtistId: str
    
    class Config:
        from_attributes = True


class RecommendationsListResponse(BaseModel):
    """
    Schema for list of recommendations
    """
    recommendations: List[RecommendationResponse]
    count: int
