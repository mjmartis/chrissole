# Produces a database of dlib embeddings for the faces
# of given indicies in images.

import face_recognition as fr
import pickle
import os
import sys

if len(sys.argv) < 3:
    print('Usage: {} input.labels output.db [list of image paths]'.format(sys.argv[0]))
    exit()

with open(sys.argv[1], 'r') as labels_file:
    labels = [int(l) for l in labels_file]

# Generate a face embedding for the given face in each image.
encs = []
for fn, l in zip(sys.argv[3:], labels):
    if l < 0:
        continue

    # Load an image some number of faces.
    image = fr.load_image_file(fn)
    
    # Generate encodings for each face in the image.
    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)
    
    # Select only the encoding for the target face.
    encs.append(face_encodings[l])

# Write the embeddings to a static database.
with open(sys.argv[2], 'wb') as db_file:
    db_file.write(pickle.dumps(encs))
