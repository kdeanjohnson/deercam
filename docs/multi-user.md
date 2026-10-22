# Multi-user DeerCam: invite-only, hosted on the web

Status: plan, 2026-10-20. Audience: a hunting club and friends, invite-only, spread across the country.

## Who's who

- **Ranch**: one place with cameras, e.g. "Home place, Garvin Co. OK" or "Club lease, KS". A ranch owns its cameras, photos, bucks, hit list, map pins, season start and region. Bucks are only ever matched within a ranch.
- **User**: one login. A user can be in several ranches (their own place plus the club lease) and switches between them in the header.
- **Member**: a user's role in one ranch.
  - *owner*: connects Reveal accounts, invites people, deletes things.
  - *member*: reviews, names and tags bucks.
  - *viewer*: looks only.
- **Reveal connection**: a Reveal login attached to a ranch. A ranch can have several, e.g. the dad's cameras plus a buddy's on the same lease. Cameras are deduplicated by camera id, so a camera shared with two members isn't imported twice.

**Why per ranch and not per user.** Club members share the same cameras and the same deer. Naming Harry's 14 once should count for everyone, and friends in other states shouldn't get matched against bucks they'll never see.

**Per member instead of per ranch:**
- phone alerts: which bucks, daylight only, quiet hours
- later, maybe a personal hit list

The hit list starts out shared by the whole ranch.

## Storage: one database per ranch

- **Each ranch** keeps today's layout in its own folder: `ranches/<id>/deercam.db`, `photos/`, `crops/` and `settings.json`.
- **A small `accounts.db`** holds users, ranches, memberships, invites, sessions, Reveal connections and AI usage.
- **Why:**
  - the matching and pattern code already assumes one data folder, so it barely changes
  - a forgotten `WHERE ranch_id=` can't leak one ranch's camera locations to another
  - backing up, exporting or deleting a ranch is a folder copy

  Postgres can come later if this ever outgrows invites.
- **The main code change:** `settings` becomes per request (a context variable set from the signed-in user's current ranch) instead of one global.
- **Volume:** your 4 cameras sent 472 photos in 31 hours, about 38 MB. Keeping full photos comes to roughly 1 GB a month for a ranch that size, which is fine on a server disk.

## Logins

- **Invites:** an owner creates an invite link and texts it. The first visit sets a password, stored hashed (argon2). Sessions use secure cookies. No email service is needed.
- **Resets:** a forgotten password gets a fresh link from an owner or the admin.
- **Every API route** checks the user's membership and role for the ranch it touches.

## Reveal logins (the sensitive part)

- **What we store:** the owner types the Reveal password once, and the server exchanges it for Cognito tokens. Only the refresh token is stored, encrypted with a key kept outside the database. The password itself is never saved.
- **When the token expires:** the owner reconnects. We'll learn how long Reveal's tokens last.
- **Risk:** this uses Reveal's private API, which can change or be blocked. Read Tactacam's terms before inviting people outside the family.

## Compute

Measured on CPU with 4 threads and no graphics card (2026-10-20). A rented server's cores are slower; budget about 2-3x.

| Step | Time |
|---|---|
| MegaDetector v6, large (`yolov10-e`, 1280 px, what we use now) | 0.85 s per photo |
| MegaDetector v6, compact (`yolov9-c`, 640 px) | 0.15 s per photo |
| DINOv2-base embedding | 0.09 s per crop |

At about 360 photos a day per ranch, even the large detector needs only a few minutes of CPU per ranch per day. Before switching to the compact model, check its accuracy against our labeled photos.

- **One worker process** with a job queue serves all ranches, with the models loaded once.
- **AI:** one key on the server, paid by the admin (decided 2026-10-20).
  - Tokens and cost are logged per ranch.
  - Each ranch has a monthly cap.
  - Owners can paste their own key instead.

## Hosting

- **Server:** one rented virtual server with a disk. Budget: $30 a month for now.
- **Docker compose:** the app, the worker and Caddy for HTTPS.
- **Domain:** a domain name.
- **Backups:** nightly, off the server (databases plus photos to object storage).
- **Self-hosting:** your PC stays an option for anyone who wants it.

## Build order

1. **Per-ranch context.** Move today's data into ranch #1. The app still runs on the PC.
2. **Accounts:** users, invite links, sessions, roles and the ranch switcher. *(Done 2026-10-21. Resets come from the admin only, since an owner resetting a password could take over an account that's also in other ranches.)*
3. **Reveal connections** per ranch, with encrypted tokens and camera dedupe.
4. **Worker:** the job queue, CPU mode and per-ranch AI usage with caps.
5. **Go live:** Docker, the server, domain and HTTPS, and backups. Invite the dad, then the club.
6. **Phone alerts** per member, as web push from the installed web app.
