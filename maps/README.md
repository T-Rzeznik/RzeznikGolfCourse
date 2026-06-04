# Hole maps

Drop your visual map for each hole here, named by hole number:

```
maps/hole_1.png
maps/hole_2.png
...
maps/hole_6.png
```

These filenames are referenced from `data/course.yaml` (the `map:` field on
each hole), so the code and notebooks can pull up the right image for a hole.

Any raster (`.png`, `.jpg`) or vector (`.svg`) format is fine — if you change the
extension, update the `map:` path in `course.yaml` to match.

Suggested things to mark on each map: tee/start, the hole/target, out-of-bounds,
trees and the house, dogleg direction, and any blind-shot lines (holes 2 and 3
have no line of sight to the hole).
