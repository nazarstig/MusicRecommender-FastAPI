from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import List


class PlaylistRequest(BaseModel):
    """
    Schema for playlist-based recommendation request
    """
    model_config = ConfigDict(populate_by_name=True)

    track_ids: List[str] = Field(
        ..., 
        validation_alias=AliasChoices("trackIds", "track_ids"),
        serialization_alias="trackIds",
        min_length=5, 
        max_length=10,
        description="Between 5 and 10 track IDs from the user's playlist"
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
