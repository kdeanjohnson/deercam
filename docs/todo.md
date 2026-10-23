# To do

Ideas and next steps that aren't built yet, roughly in order. The multi-user build order is in
[multi-user.md](multi-user.md).

## Next

- **Phone alerts** per member, as web push from the installed web app (multi-user step 6).
- **Other wildlife by name.** The AI already names non-deer animals in free text (on the home ranch so
  far: turkeys, hogs, raccoons, armadillos, skunks, grackles, crows), but the app only shows "Other
  animal". Turn that into a fixed list: turkey, hog, coyote, bobcat, raccoon, armadillo, dog, cat,
  bird, other (with the AI's words kept). Then give each its own icon and a filter on the Timeline
  and Cameras pages, and let people correct a wrong label (the AI has called things a zebra and a
  black bear here). Hogs and coyotes could get their own alerts.

## Later

- **Property lines on the map**, like onX: show parcel boundaries so a ranch's edges are easy to see,
  and maybe let an owner outline the ranch. Needs a source of parcel data; nationwide sets are paid
  (Regrid and others), and some counties and states publish theirs free. Check the license before
  showing it to other members.
- **Analyze one crop per burst** instead of every crop, to cut AI cost.
- **Auto-request HD** for buck photos (setting). Costs the camera owner; never on by default.
- **Dark mode.**
- **Tests** in the repo (today's checks live in scratch scripts).
