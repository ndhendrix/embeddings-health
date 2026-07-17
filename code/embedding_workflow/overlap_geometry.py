"""Deterministic half-stride, center-crop ownership geometry."""
from dataclasses import dataclass

@dataclass(frozen=True)
class OverlapGrid:
    chip: int
    patch: int

    def __post_init__(self):
        if self.chip % 2 or (self.chip // 2) % self.patch:
            raise ValueError("chip/2 must be divisible by patch size")

    @property
    def stride(self): return self.chip // 2
    @property
    def margin(self): return self.stride // 2
    @property
    def keep_tokens(self): return self.stride // self.patch
    @property
    def crop_token_start(self): return self.margin // self.patch

    def block_count(self, pixels: int) -> int:
        return (pixels + self.stride - 1) // self.stride

    def tile_block_bounds(self, pixels: int, tile_index: int, num_tiles: int) -> tuple[int,int]:
        count=self.block_count(pixels)
        if num_tiles < 1 or num_tiles > count or not 0 <= tile_index < num_tiles:
            raise ValueError("invalid tile partition")
        base,extra=divmod(count,num_tiles)
        start=tile_index*base+min(tile_index,extra)
        stop=start+base+(tile_index<extra)
        return start,stop

    def owned_pixel_bounds(self, pixels: int, tile_index: int, num_tiles: int) -> tuple[int,int]:
        start,stop=self.tile_block_bounds(pixels,tile_index,num_tiles)
        return start*self.stride,min(stop*self.stride,pixels)

    def chip_origin(self, block_index: int) -> int:
        return block_index*self.stride-self.margin
