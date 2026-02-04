from datetime import date
from clients.strava.client import StravaClient, StravaActivity
from database.workout_repository import WorkoutRepository
from database.athlete_repository import AthleteRepository
from models.strava_activity import StravaActivityRaw
from models.workout import ActivitySummary, DailySummary


class WorkoutAnalysisService:
    """Service for analyzing workout data from Strava."""
    
    def __init__(
        self,
        strava_client: StravaClient,
        workout_repo: WorkoutRepository,
        athlete_repo: AthleteRepository
    ):
        self.strava_client = strava_client
        self.workout_repo = workout_repo
        self.athlete_repo = athlete_repo
    
    async def get_workouts_for_date(
        self,
        athlete_id: int,
        activity_date: date,
        refresh: bool = False
    ) -> DailySummary:
        """
        Get workouts for a specific date with analysis.
        
        Uses athlete settings from database for FTP and threshold HR values.
        
        Args:
            athlete_id: Athlete ID
            activity_date: Date to get workouts for
            refresh: If True, force refresh from Strava API
            
        Returns:
            DailySummary with activities and calculated metrics
        """
        # Get athlete settings for FTP and threshold HR
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        
        # Extract FTP and threshold HR from settings
        cycling_ftp = settings.cycling.ftp if settings and settings.cycling else None
        running_ftp = settings.running.ftp if settings and settings.running else None
        threshold_hr = settings.heart_rate.lt if settings and settings.heart_rate else None
        
        activities = []
        
        if not refresh:
            # Try to load from database first
            stored_activities = await self.workout_repo.get_activities_for_date(
                athlete_id, activity_date
            )
            
            if stored_activities:
                # Convert stored data back to StravaActivity objects
                activities = [
                    StravaActivity(raw_activity.raw_data)
                    for raw_activity in stored_activities
                ]
        
        # If no cached data or refresh requested, fetch from Strava
        if not activities or refresh:
            activities = self.strava_client.get_activities_for_date(activity_date)
            
            # Store raw activities in database
            for activity in activities:
                raw_activity = StravaActivityRaw(
                    athlete_id=athlete_id,
                    activity_id=activity.id,
                    activity_date=activity_date,
                    raw_data=activity.data
                )
                await self.workout_repo.store_activity(raw_activity)
        
        # Convert to activity summaries and calculate training load
        activity_summaries = []
        for activity in activities:
            summary = ActivitySummary.from_strava_activity(activity)
            summary.calculate_training_load(
                cycling_ftp=cycling_ftp,
                running_ftp=running_ftp,
                threshold_hr=threshold_hr
            )
            activity_summaries.append(summary)
        
        # Create daily summary
        daily_summary = DailySummary(
            date=activity_date,
            activities=activity_summaries
        )
        
        return daily_summary
