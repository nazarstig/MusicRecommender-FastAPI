from pydantic import BaseModel


class SongWithArtistResponse(BaseModel):
    """
    Schema for Song response with Artist information
    """
    ArtistName: str
    ArtistId: str
    TrackId: str
    TrackName: str

    class Config:
        from_attributes = True
