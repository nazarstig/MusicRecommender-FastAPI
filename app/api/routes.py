from fastapi import APIRouter, HTTPException, Depends
from app.schemas.item import ItemResponse
from app.schemas.artist import ArtistResponse
from app.schemas.song import SongResponse
from app.models.artist import Artist
from app.models.song import Song
from app.core.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List

router = APIRouter()

@router.get("/artists", response_model=List[ArtistResponse])
async def get_artists(artistName: str, db: Session = Depends(get_db)):
    """Get first 100 artists from dbo.Artists table filtered by name starting with artistName"""
    artists = db.query(Artist).filter(Artist.ArtistName.startswith(artistName)).limit(100).all()
    return artists


@router.get("/songs", response_model=List[SongResponse])
async def get_artist_songs(artistId: str, db: Session = Depends(get_db)):
    """Get all songs for a specific artist by artist ID"""
    # Check if artist exists
    artist = db.query(Artist).filter(Artist.ArtistId == artistId).first()
    if not artist:
        raise HTTPException(status_code=404, detail=f"Artist with ID {artistId} not found")
    
    # Get all songs for the artist
    songs = db.query(Song).filter(Song.ArtistId == artistId).all()
    return songs
