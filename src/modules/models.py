from abc import ABC, abstractmethod
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, global_mean_pool, MLP

from .config import ModelConfig


class BaseGraphModel(nn.Module, ABC):
    """
    Base class for GNN models to enforce atttributes and
    methods later used in evaluation script.
    """

    def __init__(
        self,
        num_input_graph_nodes: int,  # number of unique nodes in input graphs
        num_embedding_size: int,
        num_layers: int,
        hidden: int,
        head_hidden: int | None = None,
        branch_hidden: int | None = None,
        dropout: float = 0.5,
        use_node_embeddings: bool = True,
        use_role_embeddings: bool = False,
        num_role_types: int = 0,
        role_embedding_dim: int = 0,
        use_edge_attributes: bool = False,
        use_edge_subtraction: bool = False,
        entity_class_ids: Sequence[int] | torch.Tensor | None = None,
        predicate_class_ids: Sequence[int] | torch.Tensor | None = None,
    ):
        super().__init__()
        self._num_input_graph_nodes = int(num_input_graph_nodes)
        hidden_mp = int(hidden)
        head_hidden = int(hidden if head_hidden is None else head_hidden)
        branch_hidden = int(head_hidden if branch_hidden is None else branch_hidden)
        self._dropout = float(dropout)
        self._use_node_embeddings = bool(use_node_embeddings)
        self._use_role_embeddings = bool(use_role_embeddings and num_role_types > 0 and role_embedding_dim > 0)
        self._num_role_types = int(num_role_types if self._use_role_embeddings else 0)
        self._role_embedding_dim = int(role_embedding_dim if self._use_role_embeddings else 0)
        self._mask_fill_value = float(-1e9)
        self._node_embedding_dim = int(num_embedding_size)
        input_channels = self._node_embedding_dim + (self._role_embedding_dim if self._use_role_embeddings else 0)
        self.use_edge_attributes = use_edge_attributes
        self.use_edge_subtraction = use_edge_subtraction
        assert not (self.use_edge_subtraction and not self.use_edge_attributes), (
            "Edge subtraction requires use_edge_attributes to be True."
        )

        if self.use_edge_attributes:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_mp, hidden_mp),
                nn.ReLU(),
                nn.Dropout(p=self._dropout),
                nn.Linear(hidden_mp, hidden_mp),
                nn.Dropout(p=self._dropout),
            )
        else:
            self.edge_mlp = None

        ## Initialization Step
        # NOTE: I changed this so the edge attribute handling can be more consistent.
        # self.initialization = self.create_conv_layer(input_channels, hidden_mp)
        self.initialization = nn.Sequential(
            nn.Linear(input_channels, hidden_mp),
            nn.ReLU(),
            nn.Dropout(p=self._dropout),
            # plain last layer
            nn.Linear(hidden_mp, hidden_mp),
            nn.Dropout(p=self._dropout),
        )

        # Node Embedding
        if self._use_node_embeddings:
            self.node_embeddings = nn.Embedding(
                self.num_input_graph_nodes, self._node_embedding_dim
            )
        else:
            self.node_embeddings = nn.Identity()

        if self._use_role_embeddings:
            self.role_embeddings = nn.Embedding(self._num_role_types, self._role_embedding_dim)
        else:
            self.role_embeddings = None

        # Aggregation Layers
        self.mp_layers = nn.ModuleList([self.create_conv_layer(hidden_mp, hidden_mp) for _ in range(num_layers - 1)])

        self.shared_projection = nn.Linear(hidden_mp, head_hidden)

        def make_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_hidden, branch_hidden),
                nn.ReLU(),
            )

        self.subject_branch = make_branch()
        self.object_branch = make_branch()
        self.predicate_branch = make_branch()


        self._num_target_ids = -1
        if predicate_class_ids is not None:
            self._num_target_ids = max(predicate_class_ids) + 1
        if entity_class_ids is not None:
            self._num_target_ids = max(self._num_target_ids, max(entity_class_ids) + 1)

        # If still -1, set to num_input_graph_nodes
        if self._num_target_ids == -1:
            self._num_target_ids = self.num_input_graph_nodes

        entity_ids_tensor, entity_is_full_vocab = self._prepare_class_ids(
            entity_class_ids, self.num_target_ids
        )
        predicate_ids_tensor, predicate_is_full_vocab = self._prepare_class_ids(
            predicate_class_ids, self.num_target_ids
        )
        self.register_buffer("entity_class_ids", entity_ids_tensor)
        self.register_buffer("predicate_class_ids", predicate_ids_tensor)
        self._entity_full_vocab = bool(entity_is_full_vocab)
        self._predicate_full_vocab = bool(predicate_is_full_vocab)

        entity_vocab = int(entity_ids_tensor.numel())
        predicate_vocab = int(predicate_ids_tensor.numel())

        self.subject_add_head = nn.Linear(branch_hidden, entity_vocab, bias=False)
        self.subject_del_head = nn.Linear(branch_hidden, entity_vocab, bias=False)
        self.object_add_head = nn.Linear(branch_hidden, entity_vocab, bias=False)
        self.object_del_head = nn.Linear(branch_hidden, entity_vocab, bias=False)
        self.predicate_add_head = nn.Linear(branch_hidden, predicate_vocab, bias=False)
        self.predicate_del_head = nn.Linear(branch_hidden, predicate_vocab, bias=False)
        self._hidden_channels = hidden_mp
        self._head_hidden = head_hidden
        self._branch_hidden = branch_hidden
        self._num_layers = int(num_layers)

    @property
    def num_input_graph_nodes(self) -> int:
        return self._num_input_graph_nodes
    
    @property
    def num_target_ids(self) -> int:
        return self._num_target_ids

    @property
    def hidden_channels(self) -> int:
        return self._hidden_channels

    @property
    def head_hidden(self) -> int:
        return self._head_hidden

    @property
    def branch_hidden(self) -> int:
        return self._branch_hidden

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def role_embedding_dim(self) -> int:
        return self._role_embedding_dim

    @abstractmethod
    def create_conv_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Return a torch_geometric convolution layer."""
        raise NotImplementedError

    def forward(self, data):
        # Extract data
        x, batch = data.x, data.batch
        if self.use_edge_attributes:
            edge_index = data.edge_index_non_flattened
            edge_attr = data.edge_attr_non_flattened
            # TODO: consider removing isolated nodes (requires remapping of the edge_attributes, because they currently refer to the local---in graph---node ids)
        else:
            edge_index = data.edge_index
            edge_attr = None

        if self._use_node_embeddings:
            if x.dtype not in (torch.long, torch.int64, torch.int32):
                x = x.long()
            x = self.node_embeddings(x)
        else:
            if not torch.is_floating_point(x):
                x = x.float()
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

        # Preppend role embeddings for subject/predicate/object roles
        if self._use_role_embeddings:
            role_flags = getattr(data, "role_flags", None)
            if role_flags is None:
                role_flags = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                role_flags = role_flags.view(-1)
                if role_flags.size(0) != x.size(0):
                    raise ValueError(
                        f"role_flags length {role_flags.size(0)} does not match node feature rows {x.size(0)}"
                    )
                if role_flags.dtype != torch.long:
                    role_flags = role_flags.to(dtype=torch.long)
                if role_flags.device != x.device:
                    role_flags = role_flags.to(device=x.device)
            role_features = self.role_embeddings(role_flags)
            x = torch.cat([x, role_features], dim=-1)

        # Forward pass through the model
        # x = self.initialization(x, edge_index)
        x = self.initialization(x)
        for conv in self.mp_layers:
            x = F.dropout(x, p=self._dropout, training=self.training)
            if self.use_edge_attributes:
                edge_features = self.edge_mlp(x[edge_attr])
                if self.use_edge_subtraction:
                    # TODO: Verify that this is correct, and helps the model.
                    #  An alternative implementation could learn a seperate MLP for inverse edges.
                    inverse_edge_indices = edge_index.flip(0)
                    inverse_edge_features = -edge_features
                    edge_index_forward = torch.cat([edge_index, inverse_edge_indices], dim=1)
                    edge_features_forward = torch.cat(
                        [edge_features, inverse_edge_features], dim=0
                    )
                else:
                    edge_index_forward = edge_index
                    edge_features_forward = edge_features
                x = conv(x, edge_index_forward, edge_attr=edge_features_forward)
            else:
                x = conv(x, edge_index)
        node_emb = x
        graph_emb = global_mean_pool(x, batch)

        ## Classification Head
        shared = F.relu(self.shared_projection(graph_emb))
        shared = F.dropout(shared, p=self._dropout, training=self.training)

        # Role-specificic branches for subject, object, predicate predictions
        subject_features = self.subject_branch(shared)
        subject_features = F.dropout(subject_features, p=self._dropout, training=self.training)

        object_features = self.object_branch(shared)
        object_features = F.dropout(object_features, p=self._dropout, training=self.training)

        predicate_features = self.predicate_branch(shared)
        predicate_features = F.dropout(predicate_features, p=self._dropout, training=self.training)

        # Entity logits exclude predicate IDs
        # Here we expand to prediction size (filling excluded with negative values)
        # TODO: this materializes large tensors (size=num_target_ids), though most entries never appear. Reducing the size requires adapting the training script (and evaluation), by remapping the targets.
        y_add_s = self._expand_entity_logits(self.subject_add_head(subject_features))
        y_del_s = self._expand_entity_logits(self.subject_del_head(subject_features))

        y_add_o = self._expand_entity_logits(self.object_add_head(object_features))
        y_del_o = self._expand_entity_logits(self.object_del_head(object_features))

        # Predicate logits exclude entity IDs
        y_add_p = self._expand_predicate_logits(self.predicate_add_head(predicate_features))
        y_del_p = self._expand_predicate_logits(self.predicate_del_head(predicate_features))

        prediction = torch.stack([y_add_s, y_add_p, y_add_o, y_del_s, y_del_p, y_del_o], dim=1)
        assert prediction.shape == (
            graph_emb.shape[0],
            6,
            self.num_target_ids,
        ), f"Expected {(graph_emb.shape[0], 6, self.num_target_ids)}, got {prediction.shape}"
        return {
            "edit_logits": prediction,
            "node_emb": node_emb,
            "graph_emb": graph_emb,
        }

    @staticmethod
    def _prepare_class_ids(
        class_ids: Sequence[int] | torch.Tensor | None,
        num_total_ids: int,
    ) -> tuple[torch.Tensor, bool]:
        if class_ids is None:
            tensor = torch.arange(num_total_ids, dtype=torch.long)
            return tensor, True
        if isinstance(class_ids, torch.Tensor):
            tensor = class_ids.to(dtype=torch.long)
        else:
            tensor = torch.tensor([int(idx) for idx in class_ids], dtype=torch.long)

        if tensor.numel() == 0:
            tensor = torch.tensor([0], dtype=torch.long)

        tensor = torch.unique(tensor, sorted=True)
        full_vocab = tensor.numel() == num_total_ids and torch.equal(
            tensor, torch.arange(num_total_ids, dtype=torch.long)
        )
        return tensor, full_vocab

    def _expand_entity_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self._expand_logits(logits, self.entity_class_ids, self._entity_full_vocab)

    def _expand_predicate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self._expand_logits(logits, self.predicate_class_ids, self._predicate_full_vocab)

    def _expand_logits(
        self, logits: torch.Tensor, class_ids: torch.Tensor, full_vocab: bool
    ) -> torch.Tensor:
        if full_vocab:
            return logits
        batch_size = logits.size(0)
        fill_value = torch.as_tensor(
            self._mask_fill_value, dtype=logits.dtype, device=logits.device
        )
        expanded = logits.new_full((batch_size, self.num_target_ids), fill_value)
        scatter_index = class_ids.unsqueeze(0).expand(batch_size, -1)
        expanded.scatter_(1, scatter_index, logits)
        return expanded


class RepairGAT(BaseGraphModel):
    def create_conv_layer(self, in_channels: int, out_channels: int):
        return GATConv(in_channels, out_channels, heads=1, concat=True)


class RepairGIN(BaseGraphModel):
    def create_conv_layer(self, in_channels: int, out_channels: int):
        if self.use_edge_attributes:
            # adds the edge attributes to node featrues, then uses ReLU on top.
            return GINEConv(
                Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    BN(out_channels),
                    Linear(out_channels, out_channels),
                )
            )
        else:
            return GINConv(
                Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    BN(out_channels),
                    Linear(out_channels, out_channels),
                )
            )


class RepairGCN(BaseGraphModel):
    def create_conv_layer(self, in_channels: int, out_channels: int):
        assert not self.use_edge_attributes, "GCNConv does not support edge attributes."
        return GCNConv(in_channels, out_channels, improved=True)


MODEL_REGISTRY: dict[str, Callable[..., BaseGraphModel]] = {
    "GAT": RepairGAT,
    "GIN": RepairGIN,
    "GCN": RepairGCN,
}


def build_model(model_name: str, num_input_graph_nodes: int, config: ModelConfig) -> BaseGraphModel:
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {model_name}") from exc

    return model_cls(
        num_input_graph_nodes=num_input_graph_nodes,
        num_embedding_size=config.num_embedding_size,
        num_layers=config.num_layers,
        hidden=config.hidden_channels,
        head_hidden=config.head_hidden,
        dropout=config.dropout,
        use_node_embeddings=config.use_node_embeddings,
        use_role_embeddings=config.use_role_embeddings,
        num_role_types=config.num_role_types,
        role_embedding_dim=config.role_embedding_dim,
        use_edge_attributes=config.use_edge_attributes,
        use_edge_subtraction=config.use_edge_subtraction,
        entity_class_ids=config.entity_class_ids,
        predicate_class_ids=config.predicate_class_ids,
    )


__all__ = [
    "BaseGraphModel",
    "RepairGAT",
    "RepairGIN",
    "RepairGCN",
    "build_model",
    "MODEL_REGISTRY",
]
