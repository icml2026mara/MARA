import torch

from torch import nn
from torch_harmonics import AttentionS2
from torch_scatter import scatter_add


class SEAttention(nn.Module):
    def __init__(self, input_size, hidden_size, H=4, W=8):
        """
            Initializes the model.

            Args:
                input_size (int): Size of the input.
                hidden_size (int): Size of the hidden state.
                H (int, optional): Number of grid cells along the theta (latitude) direction. Defaults to 4.
                W (int, optional): Number of grid cells along the phi (longitude) direction. Defaults to 8.
        """

        super().__init__()
        self.H = H
        self.W = W
        self.HW = H * W
        self.C = hidden_size

        # ----------------------------
        # Initial projection
        # ----------------------------
        self.reduce_input = nn.Linear(input_size, self.C)

        # ----------------------------
        # Spherical attention
        # ----------------------------
        self.dir_enc = nn.Linear(3, self.C)
        self.dist_enc = nn.Linear(1, self.C)
        self.norm_sphere = nn.LayerNorm(self.C)
        self.attn_s2 = AttentionS2(
            in_channels=self.C,
            out_channels=self.C,
            num_heads=1,
            in_shape=(H, W),
            out_shape=(H, W),
            drop_rate=0.0
        )
        self.register_buffer("sphere_grid", self.build_sphere_grid(H, W))

        # ----------------------------
        # Read-out
        # ----------------------------
        self.features_in = self.C
        self.expand_input1 = nn.Linear(H * W, 1)
        nn.init.xavier_uniform_(self.expand_input1.weight)

        self.norm_att = nn.LayerNorm(self.C)
        self.rms_node = nn.LayerNorm(self.C)

    def build_sphere_grid(self, H, W):
        """
            Builds a spherical grid.

            Args:
                H (int): Number of grid cells along the theta (latitude) direction.
                W (int): Number of grid cells along the phi (longitude) direction.

            Returns:
                torch.Tensor: Tensor of shape (H * W, 3) containing the 3D Cartesian
                coordinates of points on the unit sphere.
        """

        theta = torch.linspace(0, torch.pi, H)
        phi = torch.linspace(0, 2 * torch.pi, W)
        theta, phi = torch.meshgrid(theta, phi, indexing="ij")
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1).reshape(H * W, 3)

    def forward(self, node_feats_, position, edge_index, rbf=None):
        """
            Forward pass of the spherical attention message passing layer.
            Computes SE(3) attention over a discretized sphere.

            Args:
                node_feats_ (torch.Tensor): Input irreps of shape (N, input_size).
                position (torch.Tensor): Node 3D positions of shape (N, 3).
                edge_index (torch.Tensor): Edge indices of shape (2, E), where
                    edge_index[0] are source nodes and edge_index[1] are target nodes.
                rbf (torch.Tensor, optional): Optional radial basis function features.
                    Defaults to None.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - out: Updated node features of shape (N, input_size).
                    - gate: Gating values of shape (E, input_size).
        """

        H, W = self.H, self.W
        HW = self.HW
        device = node_feats_.device
        src, tgt = edge_index
        E = src.size(0)
        N = node_feats_.size(0)

        # ----------------------------
        # Dimensional reduction
        # ----------------------------
        node_feats = self.reduce_input(node_feats_)

        # ----------------------------
        # Spherical attention
        # ----------------------------
        grid = self.sphere_grid.to(device)
        pos_S = position[src]
        pos_T = position[tgt]
        edge_dist = (pos_S - pos_T).norm(dim=-1, keepdim=True)

        sphere_points = pos_S.unsqueeze(1) + edge_dist.unsqueeze(1) * grid.unsqueeze(0)
        dist_grid = (sphere_points - pos_T.unsqueeze(1)).norm(dim=-1, keepdim=True)

        grid_emb = self.dir_enc(grid).unsqueeze(0)
        dist_emb = self.dist_enc(dist_grid)
        sphere_feat = node_feats[tgt].unsqueeze(1) + dist_emb + grid_emb
        sphere_feat = self.norm_sphere(sphere_feat)

        sphere_feat = sphere_feat.transpose(1, 2)  # [E,C,HW]
        sphere_2d = sphere_feat.view(E, self.C, H, W)
        attn_out = self.attn_s2(sphere_2d, sphere_2d, sphere_2d)

        # ----------------------------
        # Read-out
        # ----------------------------
        attn_flat = attn_out.flatten(2).transpose(1, 2)  # [E,HW,C]
        attn_flat = self.norm_att(attn_flat).transpose(1, 2)  # [E,C,HW]
        node_feats_T = node_feats[tgt].unsqueeze(-1) + attn_flat
        node_feats_T = self.rms_node(node_feats_T.permute(0, 2, 1)).permute(0, 2, 1)

        x_rec = node_feats_T.mean(dim=1)  # [E,HW]
        x_rec = self.expand_input1(x_rec)  # [E,input_size]
        gate = torch.sigmoid(x_rec)

        # ----------------------------
        # Edge-wise message normalization
        # ----------------------------
        msg = gate * node_feats_  # [E,input_size]
        degree = scatter_add(torch.ones_like(tgt, device=device), tgt, dim=0, dim_size=N)
        msg = msg / (degree[tgt].unsqueeze(-1) + 1e-6)

        # ----------------------------
        # Residual
        # ----------------------------
        out = node_feats_.clone()
        out.index_add_(0, tgt, msg)
        out = out + node_feats_

        return out, gate
