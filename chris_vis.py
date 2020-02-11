# Process a list of images and produce copies with the different
# faces highlighted. Used to manually generate training data.

import face_recognition as fr
from PIL import Image, ImageDraw, ImageFont
import os
import sys

if len(sys.argv) < 2:
    print('Usage: {} [list of image paths]'.format(sys.argv[0]))
    exit()

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 30)

for fn in sys.argv[1:]:
    # Load an image with unknown faces.
    image = fr.load_image_file(fn)
    
    # Find all the faces in the image.
    face_locations = fr.face_locations(image)
    
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library.
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(image)
    # Create a Pillow ImageDraw Draw instance to draw with.
    draw = ImageDraw.Draw(pil_image)
    
    # Loop through each face found in the image.
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Draw a box around the face using Pillow.
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    
        # Draw a label with a name below the face.
        text_width, text_height = draw.textsize(str(i), font=font)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), str(i), fill=(255, 255, 255, 255), font=font)
    
    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    
    # Save a labeled version of the image.
    pil_image.save('{}_labeled{}'.format(*os.path.splitext(fn)))
