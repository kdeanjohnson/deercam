"""Minimal client for the Tactacam Reveal web API (the one behind account.revealcellcam.com).

Auth is AWS Cognito SRP with the web app's public pool/client ids; the API accepts the
Cognito access token as a Bearer token. Tokens are cached so we don't re-login every run.
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


def token_file():
    return settings.data_dir / ".reveal_tokens.json"


class RevealClient:
    def __init__(self, email: str | None = None, password: str | None = None):
        """With email/password: a fresh sign-in (the setup page). Without: the saved session, falling
        back to the saved password when the session has expired."""
        self._token_file = token_file()
        self._http = requests.Session()
        self._cognito = self._sign_in(email, password) if email else self._login()

    def _sign_in(self, email: str, password: str) -> Cognito:
        user = Cognito(settings.cognito_pool_id, settings.cognito_client_id,
                       user_pool_region=settings.cognito_region, username=email)
        try:
            user.authenticate(password=password)
        except Exception as e:
            raise SignInError(f"Reveal sign-in failed: {getattr(e, 'response', {}).get('Error', {}).get('Message') or e}")
        self._save(user)
        return user

    def _login(self) -> Cognito:
        if not settings.reveal_email:
            raise SignInError("Not signed in to Reveal yet; sign in on the Settings page")
        cached = json.loads(self._token_file.read_text()) if self._token_file.exists() else {}
        if cached.get("email", settings.reveal_email).lower() != settings.reveal_email.lower():
            cached = {}  # tokens belong to a different account
        user = Cognito(
            settings.cognito_pool_id,
            settings.cognito_client_id,
            user_pool_region=settings.cognito_region,
            username=settings.reveal_email,
            id_token=cached.get("id_token"),
            access_token=cached.get("access_token"),
            refresh_token=cached.get("refresh_token"),
        )
        if cached.get("refresh_token"):
            try:
                user.check_token()  # refreshes if expired
                self._save(user)
                return user
            except Exception as e:  # refresh token expired/revoked -> full login
                log.info("Cached Reveal session unusable (%s); logging in again", e)
        if not settings.reveal_password:
            raise SignInError("The Reveal sign-in has expired; sign in again on the Settings page")
        return self._sign_in(settings.reveal_email, settings.reveal_password)

    def _save(self, user: Cognito) -> None:
        self._token_file.parent.mkdir(parents=True, exist_ok=True)
        self._token_file.write_text(json.dumps({
            "email": user.username,
            "id_token": user.id_token,
            "access_token": user.access_token,
            "refresh_token": user.refresh_token,
        }))

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
        """Returns dicts with at least sharedCameraId (+ cameraName when present)."""
        # cameraType=guest: cameras others share with us (host = ones we share out)
        body = self._get("/shared-cameras", cameraType="guest")
        cams = body.get("response", body).get("cameras") or []
        if not cams or "sharedCameraId" not in cams[0]:
            raise RuntimeError(f"No shared cameras found; response keys: {list(body.get('response', body))[:20]}")
        return cams

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

    def download(self, url: str) -> bytes:
        r = self._http.get(url, timeout=120)
        r.raise_for_status()
        return r.content
