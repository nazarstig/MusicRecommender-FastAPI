from fastapi import APIRouter, HTTPException, Depends
from app.schemas.item import ItemResponse
from app.schemas.artist import ArtistResponse
from app.schemas.song import SongResponse
from app.schemas.song_with_artist import SongWithArtistResponse
from app.schemas.rating import RatingResponse
from app.schemas.recommendation import PlaylistRequest, RecommendationsListResponse, RecommendationResponse
from app.models.artist import Artist
from app.models.song import Song
from app.models.rating import Rating
from app.core.database import get_db
from app.services.recommendation import RecommendationService
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict

router = APIRouter()
recommendationService = RecommendationService()

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


@router.get("/recommendations", response_model=List[SongWithArtistResponse])
async def get_top_songs(db: Session = Depends(get_db)):
    """Get 5 songs with artist information"""
    songs_with_artists = (
        db.query(
            Artist.ArtistName,
            Artist.ArtistId,
            Song.TrackId,
            Song.TrackName
        )
        .join(Song, Artist.ArtistId == Song.ArtistId)
        .limit(5)
        .all()
    )
    
    return [
        {
            "ArtistName": row.ArtistName,
            "ArtistId": row.ArtistId,
            "TrackId": row.TrackId,
            "TrackName": row.TrackName
        }
        for row in songs_with_artists
    ]

@router.get("/recommendations/{user_id}", response_model=List[SongWithArtistResponse])
async def get_recommendations_for_user(user_id: str, db: Session = Depends(get_db)):
    """Get song recommendations for a specific user"""
    # Get recommendations for the user
    recommendations = recommendationService.recommend_songs_for_user(db, user_id, 20)
    #print(recommendations)
    print('recommendations retrieved')
    
    # Extract track IDs from recommendations
    track_ids = [rec["TrackId"] for rec in recommendations]
    
    # Query songs with artist information based on track IDs
    songs_with_artists = (
        db.query(
            Artist.ArtistName,
            Artist.ArtistId,
            Song.TrackId,
            Song.TrackName
        )
        .join(Song, Artist.ArtistId == Song.ArtistId)
        .filter(Song.TrackId.in_(track_ids))
        .all()
    )
    
    return [
        {
            "ArtistName": row.ArtistName,
            "ArtistId": row.ArtistId,
            "TrackId": row.TrackId,
            "TrackName": row.TrackName
        }
        for row in songs_with_artists
    ]


@router.get("/rated-songs/{user_id}", response_model=List[str])
async def get_rated_songs_for_user(user_id: str, db: Session = Depends(get_db)):
    """Get track IDs that the user has already rated"""
    return recommendationService.get_already_rated_songs(db, user_id)

@router.get("/createrecommendations", response_model=None)
async def create_recommendations(db: Session = Depends(get_db)):
    """Create song recommendations"""
    # Load ratings and create recommendations
    ratings = RecommendationService.loadRatings(db)
    #recommendation_service = RecommendationService()
    recommendationService.create_recommendations(ratings)
    
    return None

@router.get("/ratings", response_model=List[RatingResponse])
async def get_ratings(db: Session = Depends(get_db)):
    """Get 10 ratings from dbo.Ratings table"""
    ratings = RecommendationService.loadRatings(db)
    matrix = RecommendationService.create_user_item_matrix(ratings)
    #print(matrix.iloc[99, 99])  # 100th row and 100th column (0-indexed)
    return ratings


