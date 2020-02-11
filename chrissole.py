# Blits a rissole on top of Chris' face.

import face_recognition as fr
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
import random
import sys

# The proportion of DB faces that need to match in order to blit
# over a face in an input image.
PROP_MATCH = 0.75  

# From https://stackoverflow.com/a/14178717
# Takes a set pa 4 'dest' points and a set pb of 'source' points
# and calculates the coefficients of the affine transformation
# that maps pb to pa. 
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

if len(sys.argv) < 4:
    print('Usage: {} faces.db image.in image.out [stickers]'.format(sys.argv[0]))
    exit()

# Load the list of known faces.
with open(sys.argv[1], 'rb') as faces_file:
    faces_db = pickle.loads(faces_file.read())

# Load the target image.
image = fr.load_image_file(sys.argv[2])

# Load the sticker.
sticker = Image.open(open(random.choice(sys.argv[4:]), 'rb'))

# Find encodings for all faces in the image.
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)

pil_image = Image.fromarray(image)

# Loop through each face found in the target image.
for loc, enc in zip(face_locations, face_encodings):
    # If the face isn't likely to be our target face, skip it.
    if fr.compare_faces(faces_db, enc).count(True) < PROP_MATCH * len(faces_db):
        continue

    # See https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png 
    landmarks = fr.face_landmarks(image, face_locations=[loc])[0]

    # Get points for left and right sides of the face and chin.
    l = np.array(landmarks['chin'][0])
    r = np.array(landmarks['chin'][16])
    b = np.array(landmarks['chin'][8])

    # Project LB on to LR.
    m = ((b - l) @ (r - l)) / ((r - l) @ (r - l))

    # Get coeffs for affine transformation that maps the midpoint of each side
    # of the sticker to the sides of face and chin.
    sw, sh = sticker.size
    coeffs = find_coeffs([l, r, b, b + 2 * (l + m * (r - l) - b)],
                         [(0, sh/2), (sw, sh/2), (sw/2, sh), (sw/2, 0)])

    # Transform and blit sticker over image.
    trans_sticker = sticker.transform(pil_image.size, Image.PERSPECTIVE,
                                      coeffs, Image.BICUBIC).convert('RGBA')
    pil_image.paste(trans_sticker, (0, 0), trans_sticker)

# Save a version of the image with the sticker applied.
pil_image.save(sys.argv[3])
