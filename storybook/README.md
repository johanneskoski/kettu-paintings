# Storybook picture: the den under the oak

A watercolor-style picture painted entirely in code: glazes, gouache, ink lines and colored pencil on simulated paper.

## Files
- `akvarelli.py`: the painting machine. Paper texture, noise, masks, glazes (`glaze`), gouache, screen light, the ink pen (`Ink`, `hatch`), the colored pencil (`Pencil`) and the wash edges ("reach fields").
- `hahmot.py`: shapes, masks and ink paths of the two characters, the fox and Lumi the weasel.
- `hahmot_testi.py`: `paint_animals`, which paints the characters. It also runs alone as a quick test.
- `pesa_satu.py`: the whole picture (sky, sea, snow, oak, roots, den, animals, ink, pencil, light).

## Run
Run the commands in this folder, because the files import each other by name.

    python3 pesa_satu.py out.jpg        # full picture, 2400 x 1350 px (default name: v2.jpg)
    python3 hahmot_testi.py test.jpg    # characters only, a 660 x 400 px crop (default name: hahmot.jpg)

## Needs
- Python 3 with `numpy` and `opencv-python` (`cv2`). Nothing else.
- No input files. No photograph is used: every shape is a list of hand-placed coordinates in the code.
- A full run takes about 30 seconds (tested 22 Nov 2026), and the result matches `../images/storybook.jpg` pixel for pixel. The ink layer is drawn at 3x size (7200 x 4050 px).

## If you change things
- The coordinates assume the 2400 x 1350 canvas (`W`, `H` in `akvarelli.py`). Changing the size does not rescale the picture.
- The fox is placed with `CX`, `CY` and `SC` in `hahmot.py`. The den, lantern and tunnel are placed in `pesa_satu.py`.
