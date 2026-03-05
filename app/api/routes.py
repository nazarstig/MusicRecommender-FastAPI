from fastapi import APIRouter, HTTPException, Depends
from app.core.dependancies import get_recommendation_service
from app.schemas.artist import ArtistResponse
from app.schemas.recommendation import PlaylistRequest
from app.schemas.song import SongResponse
from app.schemas.song_with_artist import SongWithArtistResponse
from app.models.artist import Artist
from app.models.song import Song
from app.core.database import get_db
from app.services.recommendation import RecommendationService
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.get("/artists", response_model=List[ArtistResponse])
async def get_artists(artistName: str, db: Session = Depends(get_db)):
    artists = db.query(Artist).filter(Artist.ArtistName.startswith(artistName)).limit(100).all()
    return artists

@router.get("/songs", response_model=List[SongResponse])
async def get_artist_songs(artistId: str, db: Session = Depends(get_db)):
    artist = db.query(Artist).filter(Artist.ArtistId == artistId).first()
    if not artist:
        raise HTTPException(status_code=404, detail=f"Artist with ID {artistId} not found")
    
    songs = db.query(Song).filter(Song.ArtistId == artistId).all()
    return songs

@router.get("/recommendations/{user_id}", response_model=List[SongWithArtistResponse])
async def get_recommendations_for_user(
    user_id: str,
    db: Session = Depends(get_db),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    recommendations = recommendation_service.get_recommended_songs_for_user(db, user_id, 20)
    track_ids = [rec["TrackId"] for rec in recommendations]
    
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

@router.get("/createrecommendations", response_model=None)
async def create_recommendations(
    db: Session = Depends(get_db),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    recommendation_service.create_recommendations(db)
    return None

@router.post(
    "/recommendations",
    response_model=List[SongWithArtistResponse],
    responses={400: {"description": "Invalid recommendation input"}}
)
async def get_recommendations_for_tracks(
    payload: PlaylistRequest,
    db: Session = Depends(get_db),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    try:
        recommendations = recommendation_service.get_recommended_songs_from_input(payload.track_ids, 10)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    track_ids = [rec["TrackId"] for rec in recommendations]

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