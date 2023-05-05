from torch.utils.data import Sampler


class ClassSampler(Sampler):
    """
    Wraps a RandomSampler, sample a specific number
    of image to build a query set, i.e. a set that
    contains a fixed number of images of each selected
    class.
    """

    def __init__(self, cfg, dataset_metadata, selected_classes, n_query=None, shuffle=True, is_train=True, seed=3):
        self.cfg = cfg
        self.dataset_metadata = dataset_metadata
        self.selected_classes = selected_classes
        self.n_query = n_query
        self.slice = slice(0, n_query)
        self.shuffle = shuffle
        self.is_train = is_train
        self.seed = seed

        self.class_table = copy.deepcopy(dataset_metadata.class_table)
        self.filter_table()

    def filter_table(self):
        if self.is_train:
            self.class_table = filter_class_table(self.class_table, self.cfg.FEWSHOT.K_SHOT,
                                                  self.dataset_metadata.novel_classes)

    def __iter__(self):
        table = self.class_table
        selected_indices = []
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                class_id = int(c.item())
            else:
                class_id = int(c)
            keep = torch.randperm(len(table[class_id]))[self.slice]
            selected_indices = selected_indices + [table[class_id][k] for k in keep]
        selected_indices = torch.Tensor(selected_indices)
        if not self.is_train:
            selected_indices = torch.unique(selected_indices)
            # important to prevent box redundancy and low performance
        # Retrieve indices inside dataset from img ids
        self.selected_indices = selected_indices
        if self.shuffle:
            shuffle = torch.randperm(selected_indices.shape[0])
            yield from selected_indices[shuffle].long().tolist()
        else:
            yield from selected_indices.long().tolist()

    def __len__(self):
        length = 0
        all_indices = []
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                c = int(c.item())
            else:
                c = int(c)
            table = self.class_table
            all_indices = all_indices + table[c]
            if self.n_query is not None and self.n_query > 0:
                length += min(
                    self.n_query,
                    len(table[c]))
            else:
                length += len(table[c])
        if not self.is_train:
            length = torch.tensor(all_indices).unique().shape[0]
        return length


