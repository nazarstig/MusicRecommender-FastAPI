from pydantic import BaseModel


class SongResponse(BaseModel):
    """
    Schema for Song response
    """
    TrackId: str
    ArtistId: str
    TrackName: str

    class Config:
        from_attributes = True
