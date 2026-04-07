import torch


class EpisodeBank:
    def __init__(self, support_vecs, support_labels, val_features, val_labels, test_features, test_labels):
        self.support_vecs = support_vecs
        self.support_labels = support_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.test_features = test_features
        self.test_labels = test_labels

    def sample(self, split="val", class_subset=None):
        if split == "val":
            feats, labels = self.val_features, self.val_labels
        elif split == "test":
            feats, labels = self.test_features, self.test_labels
        else:
            feats, labels = self.support_vecs, self.support_labels

        if class_subset is None:
            return feats, labels

        mask = torch.zeros_like(labels).bool()
        for c in class_subset:
            mask = mask | (labels == c)
        return feats[mask], labels[mask]
