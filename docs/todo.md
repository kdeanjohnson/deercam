# To do

Ideas and next steps that aren't built yet, roughly in order. The multi-user build order is in
[multi-user.md](multi-user.md).

## Next

- **Phone alerts** per member, as web push from the installed web app (multi-user step 6).
- **Other wildlife, the rest.** Done 2026-10-23: the AI's words are sorted into a fixed list
  (`deercam/wildlife.py`), shown by name on the Timeline map and Cameras feed, filterable there, and
  members can correct a label from the map's photo panel. Still to do: an icon per animal (all use the
  paw today), wildlife counts on Home and the Engine room, and alerts for hogs and coyotes.

## Later

- **Property lines on the map**, like onX: show parcel boundaries so a ranch's edges are easy to see,
  and maybe let an owner outline the ranch. Needs a source of parcel data; nationwide sets are paid
  (Regrid and others), and some counties and states publish theirs free. Check the license before
  showing it to other members.
- **Analyze one crop per burst** instead of every crop, to cut AI cost.
- **Auto-request HD** for buck photos (setting). Costs the camera owner; never on by default.
- **Dark mode.**
- **Tests** in the repo (today's checks live in scratch scripts).
