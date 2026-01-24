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


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with database connectivity test"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {"status": "healthy", "database": db_status}


@router.get("/items", response_model=List[ItemResponse])
async def get_items():
    """Get all items"""
    # Implement your logic here
    return [
        {"id": 1, "name": "Item 1", "description": "First item"},
        {"id": 2, "name": "Item 2", "description": "Second item"}
    ]


@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get item by ID"""
    # Implement your logic here
    if item_id < 1:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {"id": item_id, "name": f"Item {item_id}", "description": f"Description for item {item_id}"}


@router.get("/artists", response_model=List[ArtistResponse])
async def get_artists(db: Session = Depends(get_db)):
    """Get first 100 artists from dbo.Artists table"""
    artists = db.query(Artist).limit(10).all()
    return artists


@router.get("/artists/{artist_id}/songs", response_model=List[SongResponse])
async def get_artist_songs(artist_id: str, db: Session = Depends(get_db)):
    """Get all songs for a specific artist by artist ID"""
    # Check if artist exists
    artist = db.query(Artist).filter(Artist.ArtistId == artist_id).first()
    if not artist:
        raise HTTPException(status_code=404, detail=f"Artist with ID {artist_id} not found")
    
    # Get all songs for the artist
    songs = db.query(Song).filter(Song.ArtistId == artist_id).all()
    return songs
