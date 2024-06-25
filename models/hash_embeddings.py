from typing import Tuple
import torch
import torch.nn as nn

# adapted from: https://github.com/daviddmc/NeSVoR/blob/master/nesvor/inr/hash_grid_torch.py


class HashEmbedder(nn.Module):
    def __init__(
        self,
        n_input_dims: int = 3,
        otype: str = "HashGrid",
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.39,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(HashEmbedder, self).__init__()
        assert n_input_dims == 3 and otype == "HashGrid"

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.b = per_level_scale

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self._get_number_of_embeddings(i), self.n_features_per_level
                )
                for i in range(n_levels)
            ]
        )
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        self.register_buffer(
            "box_offsets",
            torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]),
        )

    def _get_number_of_embeddings(self, level_idx: int) -> int:
        """
        level_idx: level index

        returns: number of embeddings for given level. Max number is 2**self.log2_hashmap_size
        """

        max_size = 2**self.log2_hashmap_size

        resolution = int(self.base_resolution * self.b**level_idx)
        n_level_size = (
            resolution + 2
        ) ** 3  # see explanation below at 'def _to_1D(...)' why we do + 2

        return min(max_size, n_level_size)

    def trilinear_interp(
        self,
        x: torch.Tensor,
        voxel_min_vertex: torch.Tensor,
        voxel_embedds: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = x - voxel_min_vertex

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 4] * weights[:, 0][:, None]
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = int(self.base_resolution * self.b**i)
            (
                voxel_min_vertex,
                hashed_voxel_indices,
                xi,
            ) = self.get_voxel_vertices(x, resolution)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(xi, voxel_min_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)

    def get_voxel_vertices(
        self, xyz: torch.Tensor, resolution: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = xyz * resolution
        voxel_min_vertex = torch.floor(xyz).int()
        voxel_indices = voxel_min_vertex.unsqueeze(1) + self.box_offsets

        max_size = 2**self.log2_hashmap_size
        n_level_size = (resolution + 2) ** 3
        if max_size > n_level_size:
            hashed_voxel_indices = self._to_1D(voxel_indices, resolution)
        else:
            hashed_voxel_indices = self._hash(voxel_indices, self.log2_hashmap_size)

        return voxel_min_vertex, hashed_voxel_indices, xyz

    def _hash(self, coords: torch.Tensor, log2_hashmap_size: int) -> torch.Tensor:
        """
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        """
        primes = [
            1,
            2654435761,
            805459861,
            3674653429,
            2097192037,
            1434869437,
            2165219737,
        ]

        xor_result = torch.zeros_like(coords)[..., 0]
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i] * primes[i]

        return (
            torch.tensor((1 << log2_hashmap_size) - 1, device=xor_result.device)
            & xor_result
        )

    def _to_1D(self, coords: torch.Tensor, resolution: int) -> torch.Tensor:
        """
        coords: 3D indices of grid
        resolution:  resolution of grid
        """

        """
        Given grid resolution, for instance 2, our coordinate values usually span from 0 to 1 (inclusive, on x, y and z dimensions).
        To convert this coordinate (which is between 0 and 1, inclusive) to a grid index,
        we multiply the coordinate with the resolution (which is 2 in this example).
        This means the maximum cell we can get is (2,2,2) when we multiply the coordinate (1,1,1) with resolution 2.
        
        If we want to convert the 3D cell index (2,2,2) into a 1D index (to retrieve the embedding),
        we can use the formula (z * resolution * resolution) + (y * resolution) + x. The resolution here however must be 3,
        since we are now dealing with a 3x3x3 grid. So, the 1D index is (2 * 3 * 3) + (2 * 3) + 2 = 26.
        
        If we use resolution 2, the 1D index would be (2 * 2 * 2) + (2 * 2) + 2 = 14. 
        This is however wrong, as it represents the wrong cell in a 3x3x3 grid.
        
        Now, we do resolution + 2 because we have offsets of + 1, so we can get a cell at location (3,3,3).
        
        """
        resolution = resolution + 2

        x = coords[:, :, 0]
        y = coords[:, :, 1]
        z = coords[:, :, 2]

        return (z * resolution * resolution) + (y * resolution) + x
