# -*- encoding:utf-8 -*-
"""
GERL 的主模型

{
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: [7, 8, 9, 10, 11],
    neighbor_news: [
        [27, 28, 29],
        [30, 31, 32],
        [33, 34, 35],
        [36, 37, 38],
        [39, 40, 41]
    ]
}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.self_attend import SelfAttendLayer
from modules.title_encoder import TitleEncoder


class Model(nn.Module):
    def __init__(self, cfg, img_embeddings=None, aggregation="additive"):
        super(Model, self).__init__()
        # Config
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.neg_count = cfg.model.neg_count
        self.embedding_size = cfg.model.id_embedding_size
        self.max_user_one_hop = cfg.dataset.max_user_one_hop
        self.max_user_two_hop = cfg.dataset.max_user_two_hop
        self.max_news_two_hop = cfg.dataset.max_news_two_hop

        self.aggregation = aggregation

        self.img_embeddings = None
        if img_embeddings is not None:
            self.img_embeddings = nn.Parameter(torch.tensor(img_embeddings, dtype=torch.float32), requires_grad=True)

        # Init Layers
        self.user_embedding = nn.Embedding(cfg.dataset.user_count, cfg.model.id_embedding_size)
        self.newsid_embedding = nn.Embedding(cfg.dataset.news_count, cfg.model.id_embedding_size)
        self.title_encoder = TitleEncoder(cfg)
        self.user_two_hop_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.user_one_hop_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.news_two_hop_id_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.news_two_hop_title_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.dropout = nn.Dropout(cfg.model.dropout)

        if img_embeddings is not None:
            self.news_two_hop_img_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
            self.user_one_hop_img_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
            self.projection_layer = nn.Linear(self.img_embeddings.shape[1], self.embedding_size)  # Project 300 to 128

        if self.aggregation == "attention":
            # Multi-Head Attention layers for aggregation
            self.user_attention = nn.MultiheadAttention(self.embedding_size, num_heads=4, batch_first=True)
            self.news_attention = nn.MultiheadAttention(self.embedding_size, num_heads=4, batch_first=True)
            self.user_projection = nn.Linear(4 * self.embedding_size if img_embeddings is not None else 3 * self.embedding_size, self.embedding_size)
            self.news_projection = nn.Linear(5 * self.embedding_size if img_embeddings is not None else 3 * self.embedding_size, self.embedding_size)

        if self.aggregation == "mlp":
            user_input_size = 4 * self.embedding_size if img_embeddings is not None else 3 * self.embedding_size
            news_input_size = 5 * self.embedding_size if img_embeddings is not None else 3 * self.embedding_size
            self.user_mlp = nn.Sequential(
                nn.Linear(user_input_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
            self.news_mlp = nn.Sequential(
                nn.Linear(news_input_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )

    def _arrange_input(self, batch):
        user, hist_news, neighbor_users = batch["user"], batch["hist_news"], batch["neighbor_users"]
        target_news, neighbor_news = batch["target_news"], batch["neighbor_news"]
        
        return user, hist_news, neighbor_users, target_news, neighbor_news

    def forward(self, user, hist_news, neighbor_users, target_news, neighbor_news, target_news_cnt):
        """
        user: [*]
        hist_news: [*, max_user_one_hop]
        neighbor_users: [*, max_user_two_hop]
        target_news: [*, target_news_cnt]
        neighbor_news: [*, target_news_cnt, max_news_two_hop]

        return [*, target_news_cnt]
        """
        user_embedding = self.user_embedding(user)
        user_embedding = self.dropout(user_embedding)
        neighbor_users_embedding = self.user_embedding(neighbor_users)
        neighbor_users_embedding = self.dropout(neighbor_users_embedding)
        neighbor_news_embedding = self.newsid_embedding(neighbor_news)
        neighbor_news_embedding = self.dropout(neighbor_news_embedding)

        # User
        user_two_hop_rep = self.user_two_hop_attend(neighbor_users_embedding)
        hist_news = hist_news.view(-1)
        hist_news_reps = self.title_encoder(hist_news)
        hist_news_reps = hist_news_reps.view(-1, self.max_user_one_hop, self.embedding_size)
        user_one_hop_rep = self.user_one_hop_attend(hist_news_reps)

        # News
        target_news = target_news.view(-1)
        target_news_reps = self.title_encoder(target_news)
        target_news_reps = target_news_reps.view(-1, target_news_cnt, self.embedding_size)
        
        neighbor_news_embedding = neighbor_news_embedding.view(-1, self.max_news_two_hop, self.embedding_size)
        news_two_hop_id_reps = self.news_two_hop_id_attend(neighbor_news_embedding)
        news_two_hop_id_reps = news_two_hop_id_reps.view(-1, target_news_cnt, self.embedding_size)

        neighbor_news = neighbor_news.view(-1)
        neighbor_news_reps = self.title_encoder(neighbor_news)
        neighbor_news_reps = neighbor_news_reps.view(-1, self.max_news_two_hop, self.embedding_size)
        news_two_hop_title_reps = self.news_two_hop_title_attend(neighbor_news_reps)
        news_two_hop_title_reps = news_two_hop_title_reps.view(-1, target_news_cnt, self.embedding_size)

        final_news_reps_list = [target_news_reps, news_two_hop_id_reps, news_two_hop_title_reps]
        final_user_reps_list = [user_one_hop_rep, user_embedding, user_two_hop_rep]
        if self.img_embeddings is not None:
            hist_imgs_reps = self.projection_layer(self.img_embeddings[hist_news])
            hist_imgs_reps = hist_imgs_reps.view(-1, self.max_user_one_hop, self.embedding_size)
            user_one_hop_img_rep = self.user_one_hop_img_attend(hist_news_reps)

            target_news_img_reps = self.projection_layer(self.img_embeddings[target_news])

            neighbor_imgs_reps = self.projection_layer(self.img_embeddings[neighbor_news.view(-1)])
            neighbor_imgs_reps = neighbor_imgs_reps.view(-1, self.max_news_two_hop, self.embedding_size)
            news_two_hop_imgs_reps = self.news_two_hop_img_attend(neighbor_imgs_reps)
            news_two_hop_imgs_reps = news_two_hop_imgs_reps.view(-1, target_news_cnt, self.embedding_size)

            final_news_reps_list.append(target_news_img_reps)
            final_news_reps_list.append(news_two_hop_imgs_reps)
            final_user_reps_list.append(user_one_hop_img_rep)

        if self.aggregation == "additive":
            final_user_rep = torch.sum(torch.stack(final_user_reps_list), dim=0)
            final_target_reps = torch.sum(torch.stack(final_news_reps_list), dim=0)
        elif self.aggregation == "attention":
            # User attention
            user_reps_stack = torch.stack(final_user_reps_list, dim=1)  # (batch_size, num_vectors, embedding_size)
            final_user_rep, _ = self.user_attention(user_reps_stack, user_reps_stack, user_reps_stack)
            final_user_rep = final_user_rep.mean(dim=1)  # Aggregate across the heads to get a single vector
            final_user_rep = self.user_projection(final_user_rep)  # Project down to embedding size

            # News attention
            news_reps_stack = torch.stack(final_news_reps_list, dim=1)  # (batch_size, num_vectors, embedding_size)
            final_target_reps, _ = self.news_attention(news_reps_stack, news_reps_stack, news_reps_stack)
            final_target_reps = final_target_reps.mean(dim=1)  # Aggregate across the heads to get a single vector
            final_target_reps = self.news_projection(final_target_reps)  # Project down to embedding size
        elif self.aggregation == "mlp":
            user_reps_concat = torch.cat(final_user_reps_list, dim=-1)  # Concatenate along the last dimension
            final_user_rep = self.user_mlp(user_reps_concat)

            news_reps_concat = torch.cat(final_news_reps_list, dim=-1)  # Concatenate along the last dimension
            final_target_reps = self.news_mlp(news_reps_concat)

        final_user_rep = final_user_rep.repeat(1, target_news_cnt).view(-1, self.embedding_size)
        final_target_reps = final_target_reps.view(-1, self.embedding_size)

        logits = torch.sum(final_user_rep * final_target_reps, dim=-1)
        logits = logits.view(-1, target_news_cnt)

        return logits

    def training_step(self, batch_data):
        # REQUIRED
        user, hist_news, neighbor_users, target_news, neighbor_news = self._arrange_input(batch_data)
        logits = self.forward(user, hist_news, neighbor_users, target_news, neighbor_news, 5)

        target = batch_data["y"]
        loss = F.cross_entropy(logits, target)

        return loss

    def prediction_step(self, batch_data, batch_size=1):
        user, hist_news, neighbor_users, target_news, neighbor_news = self._arrange_input(batch_data)
        if batch_size == 1:
            target_news = target_news.unsqueeze(-1)
            neighbor_news = neighbor_news.unsqueeze(1)
        logits = self.forward(user, hist_news, neighbor_users, target_news, neighbor_news, batch_size)
        return logits.view(-1)


class ModelDocEmb(nn.Module):
    def __init__(self, cfg, doc_embeddings=None):
        super(ModelDocEmb, self).__init__()
        # Config
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.neg_count = cfg.model.neg_count
        self.embedding_size = cfg.model.id_embedding_size
        self.max_user_one_hop = cfg.dataset.max_user_one_hop
        self.max_user_two_hop = cfg.dataset.max_user_two_hop
        self.max_news_two_hop = cfg.dataset.max_news_two_hop
        
        # self.doc_embeddings = doc_embeddings
        self.doc_embeddings = nn.Parameter(torch.tensor(doc_embeddings, dtype=torch.float32), requires_grad=True)

        print(f"Document embeddings dimension: {self.doc_embeddings.shape[1]}")
        self.projection_layer = nn.Linear(self.doc_embeddings.shape[1], self.embedding_size)  # Project 300 to 128

        # Init Layers
        self.user_embedding = nn.Embedding(cfg.dataset.user_count, cfg.model.id_embedding_size)
        self.newsid_embedding = nn.Embedding(cfg.dataset.news_count, cfg.model.id_embedding_size)
        self.user_two_hop_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.user_one_hop_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.news_two_hop_id_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.news_two_hop_title_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def _arrange_input(self, batch):
        user, hist_news, neighbor_users = batch["user"], batch["hist_news"], batch["neighbor_users"]
        target_news, neighbor_news = batch["target_news"], batch["neighbor_news"]
        
        return user, hist_news, neighbor_users, target_news, neighbor_news

    def forward(self, user, hist_news, neighbor_users, target_news, neighbor_news, target_news_cnt):
        """
        user: [*]
        hist_news: [*, max_user_one_hop]
        neighbor_users: [*, max_user_two_hop]
        target_news: [*, target_news_cnt]
        neighbor_news: [*, target_news_cnt, max_news_two_hop]

        return [*, target_news_cnt]
        """
        user_embedding = self.user_embedding(user)
        user_embedding = self.dropout(user_embedding)
        neighbor_users_embedding = self.user_embedding(neighbor_users)
        neighbor_users_embedding = self.dropout(neighbor_users_embedding)
        neighbor_news_embedding = self.newsid_embedding(neighbor_news)
        neighbor_news_embedding = self.dropout(neighbor_news_embedding)
        
        # User
        user_two_hop_rep = self.user_two_hop_attend(neighbor_users_embedding)
        hist_news_reps = self.projection_layer(self.doc_embeddings[hist_news])
        hist_news_reps = hist_news_reps.view(-1, self.max_user_one_hop, self.embedding_size)
        user_one_hop_rep = self.user_one_hop_attend(hist_news_reps)

        # News
        target_news_reps = self.projection_layer(self.doc_embeddings[target_news])
        target_news_reps = target_news_reps.view(-1, target_news_cnt, self.embedding_size)

        neighbor_news_embedding = neighbor_news_embedding.view(-1, self.max_news_two_hop, self.embedding_size)
        news_two_hop_id_reps = self.news_two_hop_id_attend(neighbor_news_embedding)
        news_two_hop_id_reps = news_two_hop_id_reps.view(-1, target_news_cnt, self.embedding_size)

        neighbor_news_reps = self.projection_layer(self.doc_embeddings[neighbor_news.view(-1)])
        neighbor_news_reps = neighbor_news_reps.view(-1, self.max_news_two_hop, self.embedding_size)
        news_two_hop_title_reps = self.news_two_hop_title_attend(neighbor_news_reps)
        news_two_hop_title_reps = news_two_hop_title_reps.view(-1, target_news_cnt, self.embedding_size)

        # Logit
        final_user_rep = user_one_hop_rep + user_embedding + user_two_hop_rep
        final_user_rep = final_user_rep.repeat(1, target_news_cnt).view(-1, self.embedding_size)
        final_target_reps = target_news_reps + news_two_hop_id_reps + news_two_hop_title_reps
        final_target_reps = final_target_reps.view(-1, self.embedding_size)

        logits = torch.sum(final_user_rep * final_target_reps, dim=-1)
        logits = logits.view(-1, target_news_cnt)

        return logits

    def training_step(self, batch_data):
        # REQUIRED
        user, hist_news, neighbor_users, target_news, neighbor_news = self._arrange_input(batch_data)
        logits = self.forward(user, hist_news, neighbor_users, target_news, neighbor_news, 5)

        target = batch_data["y"]
        loss = F.cross_entropy(logits, target)

        return loss

    def prediction_step(self, batch_data, batch_size=1):
        user, hist_news, neighbor_users, target_news, neighbor_news = self._arrange_input(batch_data)
        if batch_size == 1:
            target_news = target_news.unsqueeze(-1)
            neighbor_news = neighbor_news.unsqueeze(1)
        logits = self.forward(user, hist_news, neighbor_users, target_news, neighbor_news, batch_size)
        return logits.view(-1)
