import logging
from datetime import datetime

from analysis.engine import analyze_workout
from analysis.models import WorkoutAnalysis
from clients.strava.client import StravaClient, StravaActivity, StravaDataParser
from database.athlete_repository import AthleteRepository
from database.workout_repository import WorkoutRepository
from models.athlete import AthleteSettings
from models.strava_activity import StravaActivityRaw
from models.workout import ActivitySummary, DailySummary
from services.commute_detection import CommuteDetectionService

logger = logging.getLogger(__name__)


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
        self.commute_service = CommuteDetectionService(athlete_repo)
    
    async def get_workouts_for_date(
        self,
        athlete_id: int,
        activity_date: datetime,
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
                logger.debug(f"Loaded {len(stored_activities)} activities from cache for {activity_date.date()}")
                # Convert stored data back to StravaActivity objects
                activities = [
                    StravaActivity(raw_activity.raw_data)
                    for raw_activity in stored_activities
                ]
        
        # If no cached data or refresh requested, fetch from Strava
        if not activities or refresh:
            activities = self.strava_client.get_activities_for_date(activity_date)
            logger.debug(f"Fetched {len(activities)} activities from Strava for {activity_date.date()}")
            
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
    
    async def get_detailed_analysis(
        self,
        athlete_id: int,
        activity_id: int,
        refresh: bool = False
    ) -> WorkoutAnalysis:
        """
        Get detailed analysis for a specific activity using the new analysis engine.
        
        This provides the same rich analytics as the strava-analyze script,
        including zone distributions, lap analysis, heart rate drift, etc.
        
        Args:
            athlete_id: Athlete ID
            activity_id: Strava activity ID to analyze
            refresh: If True, force refresh from Strava API
            
        Returns:
            WorkoutAnalysis object with detailed metrics
        """
        # Get athlete settings for zone and FTP configuration
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        
        # Try to load cached analysis first (if not forcing refresh)
        # Always return cached data if available, regardless of settings changes
        if not refresh:
            cached_result = await self.workout_repo.get_analysis(athlete_id, activity_id)
            if cached_result:
                logger.debug(f"Loaded analysis for activity {activity_id} from cache")
                cached_analysis_data, _ = cached_result
                return WorkoutAnalysis(**cached_analysis_data)
        
        # Load activity, streams, and laps (from cache or API)
        activity = None
        streams = None
        laps = None
        
        if not refresh:
            # Try to load everything from database
            raw_activity = await self.workout_repo.get_activity(athlete_id, activity_id)
            if raw_activity:
                logger.debug(f"Loaded activity {activity_id} data from cache")
                activity = StravaActivity(raw_activity.raw_data)
                streams = raw_activity.streams
                laps = raw_activity.laps
        
        # Fetch from Strava if anything is missing or refresh requested
        if not activity or not streams or not laps or refresh:
            if not activity or refresh:
                logger.debug(f"Fetching activity {activity_id} from Strava")
                activity = self.strava_client.get_activity(activity_id)
                if not activity:
                    raise ValueError(f"Activity {activity_id} not found")
            
            if not streams or not laps or refresh:
                logger.debug(f"Fetching streams and laps for activity {activity_id} from Strava")
                streams_dict = self.strava_client.get_activity_streams(activity_id)
                laps = self.strava_client.get_activity_laps(activity_id)
                
                # Serialize streams to simple dict format for MongoDB storage
                streams_for_db = {stream_type: stream.data for stream_type, stream in streams_dict.items()}
                
                # Store everything in database
                raw_activity = StravaActivityRaw(
                    athlete_id=athlete_id,
                    activity_id=activity_id,
                    activity_date=activity.start_date,
                    raw_data=activity.data,
                    streams=streams_for_db,
                    laps=laps
                )
                await self.workout_repo.store_activity(raw_activity)
                
                # Use the dict[str, StravaStream] for parser
                streams = streams_dict
            else:
                # Convert cached dict back to StravaStream objects for the parser
                from clients.strava.client import StravaStream
                streams = {stream_type: StravaStream(stream_type, data) for stream_type, data in streams.items()}
        else:
            # Convert cached dict to StravaStream objects for the parser
            from clients.strava.client import StravaStream
            streams = {stream_type: StravaStream(stream_type, data) for stream_type, data in streams.items()}
        
        # Detect commute status before parsing
        is_commute, commute_name = await self.commute_service.detect_commute(
            athlete_id, activity
        )
        
        # Parse activity data WITH streams and laps
        parser = StravaDataParser(activity, streams=streams, laps=laps)
        
        # Set commute status on parser for SessionInfo
        if is_commute:
            if commute_name == "marked by athlete":
                parser.commute_status = "yes, marked by athlete"
            else:
                parser.commute_status = "yes, detected"
        else:
            parser.commute_status = "no"
        
        # Run analysis with athlete settings directly
        analysis = analyze_workout(parser, settings)
        
        # Cache the analysis results with current settings hash for reference
        analysis_dict = analysis.model_dump()
        await self.workout_repo.store_analysis(
            athlete_id, activity_id, analysis_dict, settings
        )
        
        return analysis
    
    async def get_detailed_analyses_for_date(
        self,
        athlete_id: int,
        activity_date: datetime,
        refresh: bool = False
    ) -> list[WorkoutAnalysis]:
        """
        Get detailed analyses for all activities on a specific date.
        
        Args:
            athlete_id: Athlete ID
            activity_date: Date to get workouts for
            refresh: If True, force refresh from Strava API
            
        Returns:
            List of WorkoutAnalysis objects
        """
        # Get athlete settings
        settings = await self.athlete_repo.get_athlete_settings(athlete_id)
        
        activities = []
        
        if not refresh:
            # Try to load from database first
            stored_activities = await self.workout_repo.get_activities_for_date(
                athlete_id, activity_date
            )
            
            if stored_activities:
                logger.debug(f"Loaded {len(stored_activities)} activities from cache for detailed analysis on {activity_date.date()}")
                # Convert stored data back to StravaActivity objects
                activities = [
                    StravaActivity(raw_activity.raw_data)
                    for raw_activity in stored_activities
                ]
        
        # If no cached data or refresh requested, fetch from Strava
        if not activities or refresh:
            logger.debug(f"Fetching activities from Strava for detailed analysis on {activity_date.date()}")
            activities = self.strava_client.get_activities_for_date(activity_date)
        
        # Analyze each activity using the detailed analysis method
        # This will use cached data where available
        analyses = []
        for activity in activities:
            analysis = await self.get_detailed_analysis(
                athlete_id, activity.id, refresh
            )
            analyses.append(analysis)
        
        return analyses
