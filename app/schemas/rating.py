from pydantic import BaseModel


class RatingBase(BaseModel):
    """
    Base schema for Rating
    """
    UserId: str
    TrackId: str
    Rating: int


class RatingCreate(RatingBase):
    """
    Schema for creating a new Rating
    """
    pass


class RatingResponse(RatingBase):
    """
    Schema for Rating response
    """
    
    class Config:
        from_attributes = True
