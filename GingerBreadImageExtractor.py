import argparse
from collections import defaultdict
from dataclasses import dataclass
import glob
import hashlib
import os
import sys
import struct
from typing import *

# SAVG format, UE stuff

SAVG_MAGIC = "GVAS".encode()
SAVG_HEADER_LEN_FIX = 22

@dataclass
class UEString:
    length: int
    string: str

    @classmethod
    def create(cls, bytes: bytes):
        length = struct.unpack("<I", bytes[:4])[0]
        length -= 1 # Ignore terminating zero
        string_b = struct.unpack(f"<{length}s", bytes[4:length + 4])[0]
        return cls(length, string_b.decode())
    
    def __str__(self) -> str:
        return self.string
    
    def get_serialized_size(self):
        return 4 + self.length # 4 -> length size

def is_savg_format(content: bytes) -> bool:
    "Best-effort detection for the SAVG file format"

    if len(content) < SAVG_HEADER_LEN_FIX:
        return False
    
    magic = struct.unpack("<4s", content[:4])[0]

    if magic != SAVG_MAGIC:
        return False
    
    return True

def get_savegame_folder() -> Optional[str]:
    "Try to locate the game's savegame folder. It's either installed through Steam or the Epic Games Store."

    folder_base = "%USERPROFILE%\\Saved Games\\Sackboy\\"
    folder_steam = os.path.expandvars(folder_base + "Steam")
    folder_epic = os.path.expandvars(folder_base + "Epic")
    folder_steam = os.path.join(folder_steam, "SaveGames")
    folder_epic = os.path.join(folder_epic, "SaveGames")

    if os.path.exists(folder_epic):
        return folder_epic
    elif os.path.exists(folder_steam):
        return folder_steam
    
def get_level_image_savefiles(savegame_folder: str):
    filenames = glob.glob("LevelImages_*.sav", root_dir=savegame_folder)
    return [os.path.join(savegame_folder, f) for f in filenames]

def hash_file(path: str) -> bytes:
    with open(path, 'rb', buffering=0) as f:
        return hashlib.file_digest(f, 'sha256').digest()
    
def filelist_remove_duplicates_by_content(filelist: Iterable[str]) -> list[str]:
    "This function takes a list of files and removes duplicates from the list by file content."

    # Poor man's implementation: calculate hashes, create a dict by hash, take the first entry for each hash
    files_by_hash = defaultdict(list)
    for path in filelist:
        hash = hash_file(path)
        files_by_hash[hash].append(path)

    return [paths[0] for paths in files_by_hash.values()]

def is_unreal_package_archive(data: bytes) -> bool:
    "Checks whether the stream of bytes in data look like a package file"

    return data[:4] == 0xC1832A9E.to_bytes(4)

def decompress_data(data: bytes) -> bytes:
    "Decompresses a stream of raw, compressed image data (6 gzip streams, one after the other)"

    import zlib

    result = bytes()
    leftover_data = data
    while leftover_data:
        decompressor = zlib.decompressobj()
        result += decompressor.decompress(leftover_data)

        leftover_data = decompressor.unused_data

    return result

def decode_morton_2_18bit(z: int) -> Tuple[int, int]:
    "Unpacks the original coordinates from a z-value (Morton-coded). Works for coordinates up to 511 (for efficiency)"

    # Note that there are much more efficient ways to do this. My intention here was to write code that was as easy to
    #   understand, as possible.

    assert z <= 2**18

    x = 0
    y = 0

    # From two input coordinates, Morton-coding formulates the result (z-value) by interleaving the bits of the two
    #   coordinates. This means that bits of the two numbers will follow each other in an alternating manner. So for
    #   extracting one number, we need to take every second bit, and "compact" them. For the other number,
    #   we do the very same, but starting with one position off. This is *what* we need to do.

    # This is *how* we do it:
    #   - Iterate through every bit position (0 .. 9)
    #   - Create a bitmask that selects the single bit for the position we are currently at in the loop
    #   - Take the current bit, and shift it back to its original position
    #   - Add this bit (|=) to the result

    # Further reading:
    #   - Moser–de Bruijn sequence (read this first, if you want to understand Morton coding)
    #       - https://en.wikipedia.org/wiki/Moser%E2%80%93de_Bruijn_sequence
    #   - Morton coding
    #       - https://en.wikipedia.org/wiki/Z-order_curve
    #   - Efficient decoding w/o looping, using bitwise operations only
    #       - https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

    for i in range(9):
        pos_bit_x = i * 2
        pos_bit_y = i * 2 + 1

        bit_selector_mask_x = (1 << (pos_bit_x))
        bit_selector_mask_y = (1 << (pos_bit_y))

        bit_x = (z & bit_selector_mask_x) >> (pos_bit_x)
        bit_y = (z & bit_selector_mask_y) >> (pos_bit_y)

        x |= bit_x << i
        y |= bit_y << i

    return (x, y)

def unswizzle_image(pixel_bytes: bytes, side_length: int) -> bytes:
    "Given a rectangular image, returns an arry of reordered pixels (Morton order -> row-major)"

    n_bytes = len(pixel_bytes)

    assert(n_bytes % 3 == 0)    # Pixels should be 8-bit each, w/o alpha

    result = bytearray(n_bytes)
    pixel_i = 0
    for i in range (0, n_bytes - 3, 3):
        x, y = decode_morton_2_18bit(pixel_i)

        i_1 = (x + side_length * y) * 3
        i_2 = ((x + side_length * y) * 3) + 1
        i_3 = ((x + side_length * y) * 3) + 2

        result[i_1] = pixel_bytes[i]
        result[i_2] = pixel_bytes[i + 1]
        result[i_3] = pixel_bytes[i + 2]

        pixel_i += 1

    return result

class Pixel:
    "Class representing an R8G8B8 pixel"
    def __init__(self, r: int, g: int, b: int):
        assert(r <= 256 and g <= 256 and b <= 256)
        
        self._r = r
        self._g = g
        self._b = b

    @property
    def r(self):
        return self._r
    
    @property
    def g(self):
        return self._g
    
    @property
    def b(self):
        return self._b

    def as_rgb_bytes(self) -> bytearray:
        result = bytearray()

        result.append(self._r)
        result.append(self._g)
        result.append(self._b)

        return result
    
class Image:
    "Represents an image, that is, a 2 dimensional array of pixels"

    def __init__(self, width: int, height: int, pixels: Sequence[Pixel]):
        "pixels must be a Sequence that returns Pixel objects in row-major order"
        
        self._width: int = width
        self._height: int = height

        self._pixels: List[List[Pixel]] = []   # 2D array of pixels, row-by-row (addressable as [y][x])

        for y in range(height):
            self._pixels.append([])  # Create inner list for the row

            for x in range(width):
                self._pixels[y].append(pixels[y * width + x])

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def pixels(self):
        return self._pixels

def pixels_morton_to_row_major(pixels: Sequence[Pixel], side_length: int) -> List[Pixel]:
    "Given a rectangular image, returns an arry of reordered pixels (Morton order -> row-major)"

    result = [None] * len(pixels)
    for i in range (0, len(pixels)):
        x, y = decode_morton_2_18bit(i)

        result[x + side_length * y] = pixels[i]

    return result

class BitmapFileHeader:
    SIZE = 14

    def __init__(self, file_size: int, pixels_offset: int):
        self.file_size = file_size
        self.data_offset = pixels_offset

    def serialize(self, ostream):
        header = bytearray()
        header += struct.pack("=HIHHI",
                              int.from_bytes("MB".encode()),    # Magic
                              self.file_size,                   # File size
                              0,                                # Reserved
                              0,                                # Reserved
                              self.data_offset)                 # Offset to pixel array in file
        
        assert(len(header) == BitmapFileHeader.SIZE)

        ostream.write(header)
        
class BitmapDIBHeader:
    SIZE = 40

    def __init__(self, width: int, height: int, pixel_data_size: int):
        self._width = width
        self._height = height
        self._data_size = pixel_data_size

    def serialize(self, ostream):
        header = bytearray()
        header += struct.pack("=IiiHHIIiiII",
                              BitmapDIBHeader.SIZE, # Size of this header
                              self._width,          # Width
                              self._height,         # Height
                              1,                    # Planes
                              24,                   # Format (24-bit)
                              0,                    # "Compression" (magic constant for uncompressed RGB)
                              self._data_size,      # Image data size (w/ padding)
                              0,                    # xPixelsPerMeter ("optional")
                              0,                    # yPixelsPerMeter ("optional")
                              0,                    # No. of colors in palette (unused by us)
                              0)                    # No. of "important "colors in palette (unused by us)
        
        assert(len(header) == BitmapDIBHeader.SIZE)

        ostream.write(header)

class BitmapFile:
    "In-memory representation of a bitmap file. Can be serialized into a .bmp file."

    def __init__(self, image: Image):
        # The bitmap format stores pixel data row-by-row, but the rows start from the bottom
        # Also, each row has a padding, if width % 4 != 0

        self._width = width =  image.width
        self._height = height = image.height

        n_pixels = width * height

        padding_n_per_row = width % 4   # The .bmp format has paddings for each row, to make them 4-aligned
        padding_bytes_total = padding_n_per_row * height
        row_bytes_n = width * 3 + padding_n_per_row
        
        # bytearray initializes its content to zero, we make advantage of this by not explicitly setting padding bytes to zero
        self._data_bytes = bytearray(n_pixels * 3 + padding_bytes_total)
        last_y = height - 1
        for y in range(last_y, -1, -1):
            for x in range(width):
                pixel_data_start_i = (last_y - y) * row_bytes_n + x * 3
                pixel = image._pixels[y][x]

                self._data_bytes[pixel_data_start_i] = pixel.b
                self._data_bytes[pixel_data_start_i + 1] = pixel.g
                self._data_bytes[pixel_data_start_i + 2] = pixel.r

    def serialize(self, path: str):
        # For reference, see: https://en.wikipedia.org/wiki/BMP_file_format#File_structure

        with open(path, "wb") as outfile:
            headers_size = data_offset = BitmapFileHeader.SIZE + BitmapDIBHeader.SIZE
            file_size = headers_size + len(self._data_bytes)

            file_header = BitmapFileHeader(file_size, data_offset)
            file_header.serialize(outfile)

            dib_header = BitmapDIBHeader(self._width, self._height, len(self._data_bytes))
            dib_header.serialize(outfile)

            outfile.write(self._data_bytes)

def sanitize_filename(path: str) -> str:
    DISALLOWED_CHARS = r'[<>:"|?*/\]'

    translation_table = {}
    for c in DISALLOWED_CHARS:
        translation_table[ord(c)] = "_"

    return path.translate(translation_table)

def save_bitmap_from_decompressed_image_data(data: bytes, bmp_path: str):
    "From the uncompressed merged streams comprising the (512*512) image, create and save a bitmap file"

    # Let's turn our raw bytes into pixel objects, so we can move them around and handle them with ease
    # They are in BGR format, but making Pixel instances also "implicitly" performs a BGR -> RGB conversion
    pixels_morton_order: List[Pixel] = []
    for i in range(0, len(data), 3):
        pixels_morton_order.append(Pixel(data[i + 2], # R
                                         data[i + 1], # G
                                         data[i]))    # B
        
    # Unsiwzzle the pixels, that is, reorder them from Morton order to row-major
    pixels: List[Pixel] = pixels_morton_to_row_major(pixels_morton_order, 512)

    # Create an image object
    image = Image(512, 512, pixels)

    bitmap = BitmapFile(image)
    bitmap.serialize(bmp_path)

class InplaceUnrealInlinePackage:
    def __init__(self, data: bytes, package_data_size_offset: int):
        self._raw_data_size = struct.unpack("<I", data[package_data_size_offset : package_data_size_offset + 4])[0]
        self._raw_data_start_offset = package_data_size_offset + 4
        self._raw_data_end_offset = self._raw_data_start_offset + self._raw_data_size

        if data[self._raw_data_start_offset : self._raw_data_start_offset + 4] != 0xC1832A9E.to_bytes(4):
            raise RuntimeError (f"Uknown data encountered, where an inline unreal package was expected!")

        UE_PACKAGE_HEADER_SIZE = 0x80   # Probably this is not of fixed size in general, but in our case, it is

        self._data = data[self._raw_data_start_offset + UE_PACKAGE_HEADER_SIZE : self._raw_data_end_offset]

    @property
    def data(self):
        return self._data
    
    @property
    def end_offset(self):
        return self._raw_data_end_offset
    
def decompress_eol_image(inline_package: InplaceUnrealInlinePackage) -> bytes:
    # Each image is 512x512, and pixels should be in BGR format, 8-bit color depth (3 bytes for each pixel)
    decompressed_image_data = decompress_data(inline_package.data)
    if len(decompressed_image_data) != 512 *512 * 3:
        raise RuntimeError("Invalid image data decompression result: a 512*512 (BGR) image was expected!")
    
    return decompressed_image_data

def save_eol_image(raw_image_data: bytes, image_name: UEString, outfolder: str):
    sanitized_basename = sanitize_filename(str(image_name))
    sanitized_filename = sanitized_basename + ".bmp"
    bmp_path = os.path.join(outfolder, sanitized_filename)

    if not os.path.exists(bmp_path):
        save_bitmap_from_decompressed_image_data(raw_image_data, bmp_path)

        return

    # If a level image was already extracted for that level, construct an alternative name with an ordinal
    suffix_i = 2
    while True:
        bmp_path = os.path.join(outfolder, "".join([sanitized_basename, "_", str(suffix_i), ".bmp"]))
        if os.path.exists(bmp_path):
            suffix_i += 1
        else:
            break

    save_bitmap_from_decompressed_image_data(raw_image_data, bmp_path)


def extract_images_from_sav(sav_path: str, outfolder: str) -> int:
    """Extract all end-of-level images found in the .sav file passed in sav_path to outfolder.
    Returns the number of images extracted."""

    # The savegame format of Unreal Engine is not documented, but somewhat discussed online. For a basic understanding,
    #   see this project: https://github.com/13xforever/gvas-converter

    # We don't try to actually interpret data in the file "properly", as the objects that we are interested in seem
    #   to be custom (e.g. the project linked above cannot "convert" them). Instead, we search for byte patterns,
    #   rely on fixed offsets, and so on. Brittle, not elegant, but works.

    # Since these files are about 2 MBs max., we can get away with reading their whole content at once
    with open(sav_path, "rb") as f:
        content = f.read()

    if not is_savg_format(content):
        raise RuntimeError(f"File {sav_path} does not seem to be a savegame file!")

    # Get the offset for the first "Data" property (belonging to the ImageSet)
    data_b = "Data".encode()
    first_data_offset = content.find(data_b)
    if first_data_offset == -1:
        raise RuntimeError(f"File {sav_path} does not contain data for an ImageSet object!")
    
    second_data_offset = content.find(data_b, first_data_offset + 1)
    if second_data_offset == -1:
        raise RuntimeError(f"File {sav_path} does not contain data for a Map object!")
    
    # Hack: the first level name string is always found after a fixed offset from the Map object
    first_levelname_offset = second_data_offset + 0x4A
    first_level_str = UEString.create(content[first_levelname_offset:])

    # Scan forward for the Image object, which is an array of bytes. These bytes make up an inline archive/bulk data.
    #   These seem to be "special" unreal engine package files
    first_image_offset = content.find("Image".encode(), first_levelname_offset + first_level_str.get_serialized_size())
    first_package = InplaceUnrealInlinePackage(content, first_image_offset + 0x32)

    decompressed_image_data = decompress_eol_image(first_package)
    save_eol_image(decompressed_image_data, first_level_str, outfolder)

    total_extracted = 1

    # First image is saved, iterate through the remaining images
    ue_prev_package_data_end_offset = first_package.end_offset
    while ue_prev_package_data_end_offset <= len(content) - 1:
        levelname_offset = ue_prev_package_data_end_offset + 0x37
        level_str = UEString.create(content[levelname_offset:])
        
        image_offset = content.find("Image".encode(), levelname_offset + level_str.get_serialized_size())
        if image_offset == -1:   # As the "last entry", the file might contain an image "header", but no image
            break

        package = InplaceUnrealInlinePackage(content, image_offset + 0x32)

        ue_prev_package_data_end_offset = package.end_offset
        
        decompressed_image_data = decompress_eol_image(package)
        save_eol_image(decompressed_image_data, level_str, outfolder)

        total_extracted += 1

    return total_extracted

def printprogress(completed: int, total: int):
    COMPLETION_CHAR = '█'
    INPROGRESS_CHAR = '-'
    PROBRESSBAR_LENGTH = 50

    completed_chars_n = int((completed / total) * PROBRESSBAR_LENGTH)
    inprogress_chars_n = PROBRESSBAR_LENGTH - completed_chars_n
    percent = completed / total * 100

    print(f"\rProgress: processed {completed}/{total} .sav files |{completed_chars_n * COMPLETION_CHAR}{inprogress_chars_n * INPROGRESS_CHAR}| {percent:.2f}%",
          end = "" if completed != total else "\n")
    
def extract_eol_images(savegame_folder: str, outfolder: str) -> int:
    """Extracts end-of-level images contained in savegame_folder to outfolder.
    Returns the number of unique images extracted."""

    levelimage_saves = get_level_image_savefiles(savegame_folder)
    if len(levelimage_saves) == 0:
        raise RuntimeError(f"No end of level images were found in folder {savegame_folder}!")

    # These files seem to always come in pairs, e.g. LevelImages_05.sav and LevelImages_05-1.sav
    # As some of these "duplicates" contain unique images, my suspicion is that when you beat a level with a better
    #   score and a new picture is taken, the previous one is not overwritten. Even if they contains some unique images,
    #   there might be some duplicates in them, as well. We will remove those later.
    # Either way, "full on duplicates" are not worth processing
    levelimage_saves = filelist_remove_duplicates_by_content(levelimage_saves)

    total_n_images = 0
    n_files = len(levelimage_saves)
    # For each file, extract images
    for i, f in enumerate(levelimage_saves):
        printprogress(i, n_files)

        total_n_images += extract_images_from_sav(f, outfolder)

        printprogress(i + 1, n_files)

    # Remove duplicates images by content
    image_hashes: Dict[bytes, List[str]] = defaultdict(list)
    for image_filename in os.listdir(outfolder):
        image_path = os.path.join(outfolder, image_filename)
        image_hashes[hash_file(image_path)].append(image_path)

    for hash, paths in image_hashes.items():
        if len(paths) == 1:
            continue

        # From the list of paths, remove the one with the least recent timestamp: this will be the one we keep
        timestamps = [os.path.getmtime(p) for p in paths]
        min_index = timestamps.index(min(timestamps))
        paths.pop(min_index)

        # Delete all other duplicates
        for p in paths:
            os.unlink(p)
            total_n_images -= 1

    return total_n_images

def parse_args():
    parser = argparse.ArgumentParser(
                    prog="GingerBreadImageExtractor",
                    description="This program extracts end of level images from the video game \"Sackboy: A big adventure\".",
                    epilog="https://github.com/Donpedro13/GingerBreadImageExtractor")
    
    parser.add_argument("-i", "--infolder", required = False)
    parser.add_argument("outfolder")

    return parser.parse_args()

def fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}")

    sys.exit(-1)

def main():
    args = parse_args()

    savegame_folder = get_savegame_folder()
    if not savegame_folder:
        fail("Unable to locate savegame folder. Try specifying it explicitly with the -i option.")
    else:
        print(f"Located savegame folder @ {savegame_folder}")

    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    if os.listdir(args.outfolder):
        fail("Target folder is not empty!")

    try:
        from datetime import datetime

        start_t = datetime.now()
        n_images = extract_eol_images(savegame_folder, args.outfolder)
        delta_t = datetime.now() - start_t
        
        minutes, seconds = divmod(delta_t.seconds, 60)

        print(f"Extracted {n_images} images.")
        print(f"Elapsed time: {minutes}m{seconds}s")
    except KeyboardInterrupt:
        raise
    except RuntimeError as e:
        fail(str(e))

if __name__ == "__main__":
    main()