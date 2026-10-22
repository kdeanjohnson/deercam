"""Minimal client for the Tactacam Reveal web API (the one behind account.revealcellcam.com).

Auth is AWS Cognito SRP with the web app's public pool/client ids; the API accepts the
Cognito access token as a Bearer token. The caller keeps the tokens (connections.py), so a password
is only needed to sign in the first time, or again if Reveal ends the session.
"""
import json
import logging
from typing import Iterator

import requests
from pycognito import Cognito

from .config import settings

log = logging.getLogger(__name__)


class SignInError(RuntimeError):
    """Reveal needs the user to sign in (again) on the Settings page."""


def find_location(obj) -> tuple[float, float] | None:
    """Camera GPS, if Reveal reports it: the first nested {latitude, longitude} pair that isn't 0,0.
    Only some camera models have GPS, and where it sits in the response varies."""
    if isinstance(obj, dict):
        lat, lon = obj.get("latitude"), obj.get("longitude")
        try:
            if lat is not None and lon is not None and (float(lat), float(lon)) != (0.0, 0.0):
                return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)) and (found := find_location(v)):
                return found
    return None


class RevealClient:
    def __init__(self, email: str, password: str | None = None, tokens: dict | None = None,
                 on_tokens=None, fallback_password: str | None = None):
        """With a password: a fresh sign-in. Without: the saved tokens (Cognito refreshes them as they
        expire), then fallback_password if those are gone. on_tokens(dict) is told every time the
        tokens change, so the caller can keep them (encrypted; see connections.py)."""
        self.email = email
        self._on_tokens = on_tokens or (lambda tokens: None)
        self._http = requests.Session()
        self._cognito = self._sign_in(password) if password else self._resume(tokens or {}, fallback_password)

    def _sign_in(self, password: str) -> Cognito:
        user = Cognito(settings.cognito_pool_id, settings.cognito_client_id,
                       user_pool_region=settings.cognito_region, username=self.email)
        try:
            user.authenticate(password=password)
        except Exception as e:
            raise SignInError(f"Reveal sign-in failed: {getattr(e, 'response', {}).get('Error', {}).get('Message') or e}")
        self._save(user)
        return user

    def _resume(self, tokens: dict, fallback_password: str | None) -> Cognito:
        user = Cognito(settings.cognito_pool_id, settings.cognito_client_id, user_pool_region=settings.cognito_region,
                       username=self.email, id_token=tokens.get("id_token"), access_token=tokens.get("access_token"),
                       refresh_token=tokens.get("refresh_token"))
        if tokens.get("refresh_token"):
            try:
                user.check_token()  # refreshes if expired
                self._save(user)
                return user
            except Exception as e:  # refresh token expired/revoked
                log.info("Saved Reveal session for %s unusable (%s)", self.email, e)
        if fallback_password:
            return self._sign_in(fallback_password)
        raise SignInError(f"Reveal signed out {self.email}; sign in again in Settings")

    def _save(self, user: Cognito) -> None:
        self._on_tokens({"id_token": user.id_token, "access_token": user.access_token, "refresh_token": user.refresh_token})

    def _get(self, path: str, **params) -> dict:
        if self._cognito.check_token():
            self._save(self._cognito)
        r = self._http.get(
            f"{settings.reveal_api}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._cognito.access_token}"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def shared_cameras(self) -> list[dict]:
        """Cameras others share with this account: dicts with sharedCameraId, cameraId, name and the
        owner's hostName."""
        # cameraType=guest: cameras others share with us (host = ones we share out)
        body = self._get("/shared-cameras", cameraType="guest")
        return [c for c in body.get("response", body).get("cameras") or [] if c.get("sharedCameraId")]

    def own_cameras(self) -> list[dict]:
        """Cameras on this account itself (an owner, not a guest). These carry the camera's GPS, when it
        has one; shared cameras don't."""
        body = self._get("/cameras")
        return body.get("response", body).get("cameras") or []

    def shared_photos(self, shared_camera_ids: list[str], page_size: int = 100) -> Iterator[dict]:
        """Yields photos newest-first. Each has a presigned photoUrl that expires, so download promptly."""
        page = 0  # pages are 0-indexed; page=1 skips the newest `size` photos
        while True:
            body = self._get(
                "/shared-cameras/photos",
                sharedCameraIds=",".join(shared_camera_ids),
                size=page_size,
                page=page,
            )
            photos = body["response"]["sharedCameraPhotos"]
            if not photos:
                return
            yield from photos
            page += 1

    def own_photos(self, camera_ids: list[str], page_size: int = 100) -> Iterator[dict]:
        """Photos from this account's own cameras, newest-first (what the Reveal web app's own feed
        uses). Weather may come alongside rather than inside each photo; it's attached when it does.
        Untested against a real owner account so far: the guest login we have owns no cameras."""
        key = None
        while True:
            params = {"size": page_size, "cameraIds": ",".join(camera_ids)} | ({"paginationKey": key} if key else {})
            body = self._get("/photos/v2", **params)
            r = body.get("response", body)
            weather = {w.get("id") or w.get("weatherRecordId"): w for w in r.get("weather") or [] if isinstance(w, dict)}
            photos = r.get("photos") or []
            for p in photos:
                if not p.get("weatherRecord") and p.get("weatherRecordId") in weather:
                    p["weatherRecord"] = weather[p["weatherRecordId"]]
                yield p
            key = r.get("nextToken")
            if not key or not photos:
                return

    def download(self, url: str) -> bytes:
        r = self._http.get(url, timeout=120)
        r.raise_for_status()
        return r.content
