from PIL import ImageFile

# Cellular uploads are occasionally cut short by a few bytes; decode what's there.
ImageFile.LOAD_TRUNCATED_IMAGES = True
