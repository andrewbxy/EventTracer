import numpy as np
import torch
import os
import math
from collections import OrderedDict


class TileBasedStorage:
    def __init__(
        self,
        tile_size,
        tile_count,
        directory_path,
        max_cache_size=500,
        data_format="rgba",
    ):
        self.tile_size = tile_size  # [x, y, z]
        self.tile_count = tile_count  # [x, y, z]
        self.directory_path = directory_path
        self.frame_dim = [
            tile_size[0] * tile_count[0],
            tile_size[1] * tile_count[1],
            tile_size[2] * tile_count[2],
        ]
        self.max_cache_size = max_cache_size
        self.cached_tiles = OrderedDict()
        self.data_format = data_format.lower()
        if self.data_format not in ["rgba", "int8"]:
            raise ValueError("data_format must be either 'rgba' or 'int8'")

    def _update_cache(self, key):
        """Update position of a key in cache, moving recently used item to the end"""
        if key in self.cached_tiles:
            value = self.cached_tiles.pop(key)
            self.cached_tiles[key] = value

    def _get_block_filename(self, bx, by, bz):
        return os.path.join(self.directory_path, f"block_{bx}_{by}_{bz}.bin")

    def _load_block(self, bx, by, bz):
        key = (bx, by, bz)
        filename = self._get_block_filename(bx, by, bz)

        # If in cache, update position and return
        if key in self.cached_tiles:
            self._update_cache(key)
            return self.cached_tiles[key]

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                if self.data_format == "rgba":
                    data = np.fromfile(f, dtype=np.float32)
                    # Reshape to RGBA format
                    data = data.reshape(
                        self.tile_size[2], self.tile_size[1], self.tile_size[0], 4
                    )
                elif self.data_format == "int8":
                    data = np.fromfile(f, dtype=np.int8)
                    # Reshape to single channel format
                    data = data.reshape(
                        self.tile_size[2], self.tile_size[1], self.tile_size[0], 1
                    )

                tensor = torch.from_numpy(data)

                # Check cache size, remove least recently used if full
                if len(self.cached_tiles) >= self.max_cache_size:
                    self.cached_tiles.popitem(
                        last=False
                    )  # Remove the first (oldest) item

                self.cached_tiles[key] = tensor
                return tensor
        else:
            # Return zeros if block doesn't exist
            if self.data_format == "rgba":
                return torch.zeros(
                    (self.tile_size[2], self.tile_size[1], self.tile_size[0], 4)
                )
            else:  # int8
                return torch.zeros(
                    (self.tile_size[2], self.tile_size[1], self.tile_size[0], 1),
                    dtype=torch.int8,
                )

    def get_tile(self, bx, by):
        """Get a tile at coordinates (bx, by)"""
        if not (0 <= bx < self.tile_count[0] and 0 <= by < self.tile_count[1]):
            raise IndexError(f"Tile coordinates ({bx}, {by}) out of bounds")

        result = []
        for bz in range(self.tile_count[2]):
            block = self._load_block(bx, by, bz)
            result.append(block)

        return torch.cat(result, dim=0)

    def get_voxel(self, x, y, z):
        """Get a single voxel at coordinates (x, y, z)"""
        if not (
            0 <= x < self.frame_dim[0]
            and 0 <= y < self.frame_dim[1]
            and 0 <= z < self.frame_dim[2]
        ):
            raise IndexError(f"Coordinates ({x}, {y}, {z}) out of bounds")

        bx, by, bz = (
            x // self.tile_size[0],
            y // self.tile_size[1],
            z // self.tile_size[2],
        )
        lx, ly, lz = x % self.tile_size[0], y % self.tile_size[1], z % self.tile_size[2]

        block = self._load_block(bx, by, bz)
        return block[lz, ly, lx]

    def get(self, x=None, y=None, z=None):
        """Dynamic accessor based on specified dimensions"""
        if x is not None and y is not None and z is not None:
            # Single voxel
            return self.get_voxel(x, y, z)

        elif x is not None and y is not None:
            # Get a column (z-line)
            result = []
            for bz in range(self.tile_count[2]):
                bx, by = x // self.tile_size[0], y // self.tile_size[1]
                lx, ly = x % self.tile_size[0], y % self.tile_size[1]
                block = self._load_block(bx, by, bz)
                column = block[:, ly, lx]
                result.append(column)
            return torch.cat(result, dim=0)

        elif x is not None and z is not None:
            # Get a row (y-line)
            result = []
            for by in range(self.tile_count[1]):
                bx, bz = x // self.tile_size[0], z // self.tile_size[2]
                lx, lz = x % self.tile_size[0], z % self.tile_size[2]
                block = self._load_block(bx, by, bz)
                row = block[lz, :, lx]
                result.append(row)
            return torch.cat(result, dim=0)

        elif y is not None and z is not None:
            # Get x-line
            result = []
            for bx in range(self.tile_count[0]):
                by, bz = y // self.tile_size[1], z // self.tile_size[2]
                ly, lz = y % self.tile_size[1], z % self.tile_size[2]
                block = self._load_block(bx, by, bz)
                line = block[lz, ly, :]
                result.append(line)
            return torch.cat(result, dim=0)

        elif x is not None:
            # Get YZ plane
            result = torch.zeros(
                (
                    self.frame_dim[2],
                    self.frame_dim[1],
                    self.data_format == "rgba" and 4 or 1,
                )
            )
            bx = x // self.tile_size[0]
            lx = x % self.tile_size[0]

            for by in range(self.tile_count[1]):
                for bz in range(self.tile_count[2]):
                    block = self._load_block(bx, by, bz)
                    plane_part = block[:, :, lx]

                    y_start = by * self.tile_size[1]
                    z_start = bz * self.tile_size[2]
                    y_end = min((by + 1) * self.tile_size[1], self.frame_dim[1])
                    z_end = min((bz + 1) * self.tile_size[2], self.frame_dim[2])

                    result[z_start:z_end, y_start:y_end] = plane_part[
                        : z_end - z_start, : y_end - y_start
                    ]

            return result

        elif y is not None:
            # Get XZ plane
            result = torch.zeros(
                (
                    self.frame_dim[2],
                    self.frame_dim[0],
                    self.data_format == "rgba" and 4 or 1,
                )
            )
            by = y // self.tile_size[1]
            ly = y % self.tile_size[1]

            for bx in range(self.tile_count[0]):
                for bz in range(self.tile_count[2]):
                    block = self._load_block(bx, by, bz)
                    plane_part = block[:, ly, :]

                    x_start = bx * self.tile_size[0]
                    z_start = bz * self.tile_size[2]
                    x_end = min((bx + 1) * self.tile_size[0], self.frame_dim[0])
                    z_end = min((bz + 1) * self.tile_size[2], self.frame_dim[2])

                    result[z_start:z_end, x_start:x_end] = plane_part[
                        : z_end - z_start, : x_end - x_start
                    ]

            return result

        elif z is not None:
            # Get XY plane (frame)
            result = torch.zeros(
                (
                    self.frame_dim[1],
                    self.frame_dim[0],
                    self.data_format == "rgba" and 4 or 1,
                )
            )
            bz = z // self.tile_size[2]
            lz = z % self.tile_size[2]

            for bx in range(self.tile_count[0]):
                for by in range(self.tile_count[1]):
                    block = self._load_block(bx, by, bz)
                    plane_part = block[lz, :, :]

                    x_start = bx * self.tile_size[0]
                    y_start = by * self.tile_size[1]
                    x_end = min((bx + 1) * self.tile_size[0], self.frame_dim[0])
                    y_end = min((by + 1) * self.tile_size[1], self.frame_dim[1])

                    result[y_start:y_end, x_start:x_end] = plane_part[
                        : y_end - y_start, : x_end - x_start
                    ]

            return result[:720, :1280, :]

        else:
            # If all are None, we would need to load the entire volume
            raise ValueError("At least one dimension must be specified")

    def clear_cache(self):
        """Clear the entire cache"""
        self.cached_tiles.clear()

    def get_cache_info(self):
        """Return cache information"""
        return {"current_size": len(self.cached_tiles), "max_size": self.max_cache_size}


def lin_log(rgb, threshold=0.1):
    # converting rgb into np.float64.
    if rgb.dtype is not torch.float64:  # note float64 to get rounding to work
        rgb = rgb.double()

    assert rgb.shape == (720, 1280, 4)
    x = (
        rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    )  # RGB to illumination

    f = (1.0 / threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x * f, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = 1e8
    y = torch.round(y * rounding) / rounding

    return y.float()

def block_lin_log(rgb, type):
    # converting rgb into np.float64.
    if rgb.dtype is not torch.float64:  # note float64 to get rounding to work
        rgb = rgb.double()

    if type == "lin_log":
        threshold = 20
    elif type == "log":
        threshold = 3
    elif type == "ldr":
        threshold = 20
        gamma = 1/2.2  # Standard gamma correction value
        rgb = np.power(rgb, gamma)
        rgb = np.clip(rgb, 0.0, 1.0)

    x = (
        rgb[:, :, :, 0] * 0.2126 + rgb[:, :, :, 1] * 0.7152 + rgb[:, :, :, 2] * 0.0722
    ) * 255  # RGB to illumination

    f = (1.0 / threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x * f, torch.log(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    rounding = 1e8
    y = torch.round(y * rounding) / rounding

    return y.float()