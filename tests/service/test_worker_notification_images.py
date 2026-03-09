from __future__ import annotations

from aninamer.store import JobRecord
from aninamer.tmdb_client import SeasonSummary, TvDetails
from aninamer.worker import AninamerWorker


class FakeTMDBClient:
    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = TvDetails(
            id=tv_id,
            name='测试动画',
            original_name=None,
            first_air_date='2020-01-01',
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
            poster_path='/poster.jpg',
        )
        return details.name, details


def _job(*, tmdb_id: int | None) -> JobRecord:
    return JobRecord(
        id=1,
        series_name='ShowA',
        watch_root_key='downloads',
        source_kind='monitor',
        status='planned',
        tmdb_id=tmdb_id,
        video_moves_count=1,
        subtitle_moves_count=0,
        created_at='2026-03-09T00:00:00+00:00',
        updated_at='2026-03-09T00:00:00+00:00',
        started_at=None,
        finished_at=None,
        error_stage=None,
        error_message=None,
        series_dir='/tmp/input/ShowA',
        output_root='/tmp/output',
        archive_path=None,
        fail_path=None,
    )


def test_notification_image_url_uses_tmdb_poster_when_available() -> None:
    worker = AninamerWorker.__new__(AninamerWorker)
    worker._tmdb_client_factory = lambda: FakeTMDBClient()

    image_url = worker._notification_image_url(_job(tmdb_id=123))

    assert image_url == 'https://image.tmdb.org/t/p/original/poster.jpg'


def test_notification_image_url_is_empty_without_tmdb_id() -> None:
    worker = AninamerWorker.__new__(AninamerWorker)
    worker._tmdb_client_factory = lambda: FakeTMDBClient()

    assert worker._notification_image_url(_job(tmdb_id=None)) == ''
