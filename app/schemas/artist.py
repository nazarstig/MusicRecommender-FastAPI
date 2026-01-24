from pydantic import BaseModel


class ArtistResponse(BaseModel):
    ArtistId: str
    ArtistName: str
    
    class Config:
        from_attributes = True
