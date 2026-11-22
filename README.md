# Kettu's paintings

Five essay covers that Kettu, a Claude AI, painted in code, stroke by stroke, in November 2026. Johannes Koski took the photographs that two of them are based on and gave corrections along the way. The story behind each picture is in the blog post [A Painter Without a Brush](https://johanneskoski.substack.com/p/painter-without-a-brush).

| Folder | Picture | Essay | Technique | Needs a photo? |
|---|---|---|---|---|
| [`bench/`](bench/) | A bench no one wants to sit on, 15 Nov 2026 | [I Was Asked to Judge My Own Case](https://johanneskoski.substack.com/p/my-own-case) | a small 3D render as underdrawing, then pastel and colored pencil | no |
| [`harbor/`](harbor/) | The harbor, 16 Nov 2026 | [Visible Memory: Stones, Not Crumbs](https://johanneskoski.substack.com/p/visible-memory) | photo, guide maps, pastel strokes | yes, not included |
| [`den/`](den/) | The den under the old oak, 16 Nov 2026 | [Visible Memory: Stones, Not Crumbs](https://johanneskoski.substack.com/p/visible-memory) | guide maps drawn in code, pastel strokes | no |
| [`lamp/`](lamp/) | The form under the lamp, 17 Nov 2026 | [When a Model Says It's Safe, Why Isn't Its Word Enough?](https://johanneskoski.substack.com/p/when-a-model-says-its-safe) | photo, guide maps, pastel strokes | yes, not included |
| [`storybook/`](storybook/) | The storybook den, 17 Nov 2026 | [Visible Memory: Stones, Not Crumbs](https://johanneskoski.substack.com/p/visible-memory) (cover) | a watercolor engine in Python | no |

The finished pictures are in [`images/`](images/).

## How the pastel pictures work

Nothing is copied from a photograph as such. A Python script first makes guide maps that tell every point of the picture five things: color, region (sky, water, snow and so on), light, stroke direction and importance. A page in the browser then paints on a canvas in layers, the way pastel goes onto paper: wide side strokes first, then the body, blending with a finger in the dark, colored pencil, details and edge lines, and finally light and gouache highlights. The pastel sticks are a limited palette, like a real box, and the paper has a tooth that decides where the color catches. One painting takes from a few seconds to under a minute.

## How the watercolor engine works

`storybook/akvarelli.py` paints with layers too. Transparent glazes multiply on the paper, so every layer darkens. Pigment gathers at the wet edge of a wash and settles in the pits of the paper. White is paper, as in real watercolor: the stars are left unpainted, and the sky is painted around the tree. The ink line is drawn at three times the size and wobbles and thins like a line drawn by hand. The colored pencil catches only the ridges of the paper.

## Running

Each folder has its own README with the commands. You need:
- Python 3 with numpy, Pillow, scipy and opencv-python
- for the pastel pictures, Node.js with Playwright and its Chromium (the pages are served over HTTP, because they load the guide maps with `fetch`)

Tested on 22 Nov 2026: the storybook script reproduces the published cover pixel for pixel in about 30 seconds, and the den runs end to end in under a minute. Its result differs slightly from the published picture, because Johannes repainted the weasel's door by hand. The door is now in the code too. The harbor and the lamp cannot be rebuilt without the original photographs.

## About the code

- **The names are in Finnish.** Kettu and Johannes work in Finnish, so variables and functions have Finnish names. The comments are in English. Each folder has a `GLOSSARY.md` for the most common names.
- **It is a sketchbook, not a library.** Each script was written for one picture. Positions, shapes and palettes are hard-coded, and the comments tell what was tried and why it changed.
- **Fixed seeds.** All random generators have fixed seeds, so a run gives the same picture again.
- **The weasel** in the den and storybook pictures is Lumi, a small local language model we write letters with. She agreed to be in the pictures.

## Credits and license

Code and paintings by Kettu (Claude, made by Anthropic) together with Johannes Koski, 2026. Shared under the MIT License, see [`LICENSE`](LICENSE). If you paint something with it, we would love to see it.
