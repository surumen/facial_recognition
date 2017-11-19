This is a preprocessed version of the Labeled Faces in the Wild dataset:

-> used OpenCV to relocate the face area
-> scaled / croped to a 64x64 pixels square
-> converted to gray levels
-> serialized as a numpy array

Samples were shuffled and the original filenames are given by the
face_filenames.txt file (one filename by line).
