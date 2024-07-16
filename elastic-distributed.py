import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing import run

print('Helloooo!!')

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]
        print('Dataset is being created')

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

def ddp_setup(backend="gloo"):
    print('DDP is being set up...')
    init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else torch.device('cpu')
    print(local_rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}" if torch.cuda.is_available() else 'cpu'
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        print('Starting the training...')
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main_worker(local_rank, args):
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    print('Setting up DDP...')
    ddp_setup(backend=backend)
    print('DDP is set. Setting up the model, optimizer and dataset')
    dataset, model, optimizer = load_train_objs()
    print('Prepping the dataloader')
    train_data = prepare_dataloader(dataset, args.batch_size)
    trainer = Trainer(model, train_data, optimizer, args.save_every, args.snapshot_path)
    print('Training is setting up...')
    trainer.train(args.total_epochs)
    destroy_process_group()
    print('Training is done, destroyed the process group')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple distributed training job')
    parser.add_argument('total_epochs', default=30, type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', default=0, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--snapshot_path', default='snapshot.pt', type=str, help='Path to save the snapshot')
    parser.add_argument('--min_size', default=1, type=int, help='Minimum number of nodes')
    parser.add_argument('--max_size', default=4, type=int, help='Maximum number of nodes')
    args = parser.parse_args()
    print('Arguments are passed')

    run(
        main_worker,
        args=(args,),
        nprocs_per_node=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        min_size=args.min_size,
        max_size=args.max_size,
        start_method='spawn'
    )

if __name__ == "__main__":
    main()