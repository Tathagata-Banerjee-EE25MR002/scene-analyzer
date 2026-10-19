"""Accounts and usage storage. SQLite for local use; any Postgres through DATABASE_URL in production."""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.engine import Engine

metadata = sa.MetaData()

users = sa.Table(
    "users", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("username", sa.String(32), nullable=False, unique=True),  # stored lower-case
    sa.Column("password_hash", sa.String(255), nullable=False),
    sa.Column("is_admin", sa.Boolean, nullable=False, default=False),
    sa.Column("is_active", sa.Boolean, nullable=False, default=True),
    sa.Column("session_version", sa.Integer, nullable=False, default=1),  # bump to sign the user out everywhere
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("last_login_at", sa.DateTime, nullable=True),
)

usage = sa.Table(
    "usage", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("day", sa.String(10), nullable=False),  # UTC date, YYYY-MM-DD
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("input_tokens", sa.Integer, nullable=False, default=0),
    sa.Column("output_tokens", sa.Integer, nullable=False, default=0),
)
sa.Index("ix_usage_user_day", usage.c.user_id, usage.c.day)
sa.Index("ix_usage_day", usage.c.day)


def normalize_url(url: str) -> str:
    """Accept the forms hosts hand out (postgres://, postgresql://) and pick the psycopg 3 driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def today_utc() -> str:
    return utc_now().strftime("%Y-%m-%d")


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_admin: bool
    is_active: bool
    session_version: int
    password_hash: str = ""


def _user(row) -> User:
    return User(row.id, row.username, bool(row.is_admin), bool(row.is_active), row.session_version, row.password_hash)


class UsernameTaken(Exception):
    pass


class Database:
    def __init__(self, url: str):
        url = normalize_url(url)
        kwargs: dict = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            path = url.split("///", 1)[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs.update(pool_size=5, max_overflow=5, pool_recycle=1800)
        self.engine: Engine = sa.create_engine(url, **kwargs)
        self.dialect = self.engine.dialect.name
        self._quota_lock = threading.Lock()
        metadata.create_all(self.engine)

    # ------------------------------------------------------------------ users

    def create_user(self, username: str, password_hash: str, is_admin: bool = False) -> User:
        try:
            with self.engine.begin() as c:
                res = c.execute(users.insert().values(
                    username=username.lower(), password_hash=password_hash, is_admin=is_admin,
                    is_active=True, session_version=1, created_at=utc_now()))
                row = c.execute(users.select().where(users.c.id == res.inserted_primary_key[0])).one()
        except sa.exc.IntegrityError as exc:
            raise UsernameTaken(username) from exc
        return _user(row)

    def get_user(self, user_id: int) -> User | None:
        with self.engine.connect() as c:
            row = c.execute(users.select().where(users.c.id == user_id)).first()
        return _user(row) if row else None

    def get_user_by_name(self, username: str) -> User | None:
        with self.engine.connect() as c:
            row = c.execute(users.select().where(users.c.username == username.lower())).first()
        return _user(row) if row else None

    def touch_login(self, user_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(users.update().where(users.c.id == user_id).values(last_login_at=utc_now()))

    def set_password(self, user_id: int, password_hash: str) -> int:
        """Store a new password and sign the user out everywhere. Returns the new session version."""
        with self.engine.begin() as c:
            c.execute(users.update().where(users.c.id == user_id).values(
                password_hash=password_hash, session_version=users.c.session_version + 1))
            return c.execute(sa.select(users.c.session_version).where(users.c.id == user_id)).scalar_one()

    def set_active(self, user_id: int, active: bool) -> None:
        with self.engine.begin() as c:
            c.execute(users.update().where(users.c.id == user_id).values(
                is_active=active, session_version=users.c.session_version + 1))

    def set_admin(self, user_id: int, admin: bool) -> None:
        with self.engine.begin() as c:
            c.execute(users.update().where(users.c.id == user_id).values(is_admin=admin, is_active=True))

    def delete_user(self, user_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(usage.delete().where(usage.c.user_id == user_id))  # explicit: SQLite ignores ON DELETE CASCADE by default
            c.execute(users.delete().where(users.c.id == user_id))

    def list_users(self) -> list[dict]:
        today = today_utc()
        total = sa.func.count(usage.c.id).label("total")
        today_n = sa.func.coalesce(sa.func.sum(sa.case((usage.c.day == today, 1), else_=0)), 0).label("today")
        q = (sa.select(users.c.id, users.c.username, users.c.is_admin, users.c.is_active, users.c.created_at,
                       users.c.last_login_at, total, today_n)
             .select_from(users.outerjoin(usage, usage.c.user_id == users.c.id))
             .group_by(users.c.id).order_by(users.c.created_at.desc()))
        with self.engine.connect() as c:
            return [
                {"id": r.id, "username": r.username, "is_admin": bool(r.is_admin), "is_active": bool(r.is_active),
                 "created_at": r.created_at.isoformat() + "Z", "last_login_at": r.last_login_at.isoformat() + "Z" if r.last_login_at else None,
                 "used_today": int(r.today), "total_used": int(r.total)}
                for r in c.execute(q)
            ]

    def count_users(self) -> int:
        with self.engine.connect() as c:
            return c.execute(sa.select(sa.func.count()).select_from(users)).scalar_one()

    # ------------------------------------------------------------------ usage and quotas

    def used_today(self, user_id: int) -> int:
        with self.engine.connect() as c:
            return c.execute(sa.select(sa.func.count()).select_from(usage).where(
                usage.c.user_id == user_id, usage.c.day == today_utc())).scalar_one()

    def reserve(self, user_id: int, user_limit: int, global_limit: int) -> tuple[int | None, str | None]:
        """Take one analysis from the user's and the global daily allowance.

        Returns (usage_id, None) on success or (None, "user" | "global") naming the limit that was hit.
        The lock keeps two requests on one instance from both squeezing under the limit.
        """
        day = today_utc()
        with self._quota_lock, self.engine.begin() as c:
            if user_limit > 0:
                n = c.execute(sa.select(sa.func.count()).select_from(usage).where(
                    usage.c.user_id == user_id, usage.c.day == day)).scalar_one()
                if n >= user_limit:
                    return None, "user"
            if global_limit > 0:
                n = c.execute(sa.select(sa.func.count()).select_from(usage).where(usage.c.day == day)).scalar_one()
                if n >= global_limit:
                    return None, "global"
            res = c.execute(usage.insert().values(user_id=user_id, day=day, created_at=utc_now()))
            return res.inserted_primary_key[0], None

    def finish(self, usage_id: int, input_tokens: int, output_tokens: int) -> None:
        with self.engine.begin() as c:
            c.execute(usage.update().where(usage.c.id == usage_id).values(
                input_tokens=input_tokens, output_tokens=output_tokens))

    def refund(self, usage_id: int) -> None:
        with self.engine.begin() as c:
            c.execute(usage.delete().where(usage.c.id == usage_id))

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(sa.text("SELECT 1"))
            return True
        except Exception:
            return False
